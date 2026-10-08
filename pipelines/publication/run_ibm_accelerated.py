"""Resumable accelerated IBM hardware execution.

Uses the existing frozen QPY and original record indices.
Never modifies the original plan or previously collected batches.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

ROOT = Path("results/publication/ibm_hardware_validation")
ACCEL = ROOT / "accelerated"
PLAN = ROOT / "resumable_plan.json"
QPY = ROOT / "frozen_circuits.qpy"
JOBS = ACCEL / "jobs.json"

INSTANCE = "hqnn-thesis-instance"
BACKEND = "ibm_marrakesh"
SHOTS = 1024

RANGES = [
    (75, 375),
    (375, 675),
    (675, 975),
    (975, 1275),
    (1275, 1500),
]


def load(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def verified_plan():
    plan = load(PLAN)
    if plan is None:
        raise RuntimeError("Missing frozen plan.")

    digest = hashlib.sha256(QPY.read_bytes()).hexdigest()

    if digest != plan["qpy_sha256"]:
        raise RuntimeError("Frozen QPY checksum mismatch.")

    if plan["circuit_count"] != 1500:
        raise RuntimeError("Unexpected circuit count.")

    if len(plan["records"]) != 1500:
        raise RuntimeError("Unexpected record count.")

    return plan


def preflight():
    plan = verified_plan()

    original_jobs = load(ROOT / "jobs.json", {})

    for batch in range(5):
        info = original_jobs.get(str(batch))

        if info is None or info["status"] != "COLLECTED":
            raise RuntimeError(
                f"Original batch {batch} is not collected."
            )

        if (info["start"], info["end"]) != (
            batch * 15, (batch + 1) * 15
        ):
            raise RuntimeError("Original batch range mismatch.")

        path = ROOT / f"batch_{batch:03d}_counts.json"
        data = load(path)

        if data is None:
            raise RuntimeError(f"Missing {path}")

        if data["job_id"] != info["job_id"]:
            raise RuntimeError("Original job ID mismatch.")

        if data["qpy_sha256"] != plan["qpy_sha256"]:
            raise RuntimeError("Original QPY checksum mismatch.")

        if len(data["records"]) != 15:
            raise RuntimeError("Original batch length mismatch.")

        for i, row in enumerate(data["records"]):
            expected = plan["records"][info["start"] + i]

            for field in ("seed", "split", "sample", "basis"):
                if row[field] != expected[field]:
                    raise RuntimeError(
                        "Original measurement mapping mismatch."
                    )

            counts = row["counts"]

            if sum(counts.values()) != SHOTS:
                raise RuntimeError("Original shot mismatch.")

            if any(len(k) != 4 for k in counts):
                raise RuntimeError("Invalid original bitstring.")

    print("Frozen QPY verified.")
    print("Original 75 circuits verified.")
    print("Accelerated ranges:", RANGES)
    print("No hardware jobs submitted.")


def submit(batch, confirm):
    plan = verified_plan()
    preflight()

    if not confirm:
        print("Submission disabled without --confirm-submit.")
        return

    if batch < 0 or batch >= len(RANGES):
        raise ValueError("Invalid accelerated batch.")

    start, end = RANGES[batch]
    key = str(batch)

    jobs = load(JOBS, {})

    if key in jobs:
        raise RuntimeError(
            "Batch already recorded. Do not resubmit."
        )

    with QPY.open("rb") as f:
        circuits = qpy.load(f)

    if len(circuits) != 1500:
        raise RuntimeError("Frozen circuit count mismatch.")

    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)

    if not backend.status().operational:
        raise RuntimeError("Backend is not operational.")

    sampler = SamplerV2(mode=backend)

    jobs[key] = {
        "status": "SUBMISSION_ATTEMPTED",
        "start": start,
        "end": end,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qpy_sha256": plan["qpy_sha256"],
    }

    save(JOBS, jobs)

    print(
        f"Submitting accelerated batch {batch}: "
        f"circuits [{start}, {end})",
        flush=True,
    )

    try:
        job = sampler.run(
            circuits[start:end],
            shots=SHOTS,
        )
    except Exception as exc:
        jobs[key]["status"] = "SUBMISSION_UNCERTAIN"
        jobs[key]["error"] = str(exc)
        save(JOBS, jobs)
        raise RuntimeError(
            "Submission uncertain. Check IBM Workloads "
            "before any retry."
        ) from exc

    jobs[key]["status"] = "SUBMITTED"
    jobs[key]["job_id"] = job.job_id()
    save(JOBS, jobs)

    print("IBM Job ID:", job.job_id())


def collect(batch):
    plan = verified_plan()

    jobs = load(JOBS, {})
    key = str(batch)

    if key not in jobs:
        raise RuntimeError("Batch has not been submitted.")

    info = jobs[key]
    output = ACCEL / f"batch_{batch:03d}_counts.json"

    if output.exists():
        data = load(output)

        if (
            data["job_id"] != info.get("job_id")
            or data["qpy_sha256"] != plan["qpy_sha256"]
        ):
            raise RuntimeError("Existing output mismatch.")

        if len(data["records"]) != info["end"] - info["start"]:
            raise RuntimeError("Existing record count mismatch.")

        for i, row in enumerate(data["records"]):
            expected = plan["records"][info["start"] + i]
            for field in ("seed", "split", "sample", "basis"):
                if row[field] != expected[field]:
                    raise RuntimeError("Existing record mapping mismatch.")
            if sum(row["counts"].values()) != SHOTS:
                raise RuntimeError("Existing shot count mismatch.")
            if any(len(k) != 4 for k in row["counts"]):
                raise RuntimeError("Existing bitstring mismatch.")

        if info["status"] != "COLLECTED":
            raise RuntimeError("Existing output has inconsistent job state.")

        print("Batch already collected and verified.")
        return

    if not info.get("job_id"):
        raise RuntimeError(
            "Missing job ID. Resolve in IBM Workloads."
        )

    service = QiskitRuntimeService(instance=INSTANCE)
    job = service.job(info["job_id"])

    status = str(job.status()).upper()
    print("Job:", info["job_id"], "Status:", status)

    if status not in ("DONE", "COMPLETED"):
        print("Not completed. No data collected.")
        return

    result = job.result()

    rows = []

    for offset, pub in enumerate(result):
        index = info["start"] + offset

        if index >= info["end"]:
            raise RuntimeError("Unexpected extra result.")

        counts = pub.data.meas.get_counts()

        if sum(counts.values()) != SHOTS:
            raise RuntimeError("Shot count mismatch.")

        if any(len(k) != 4 for k in counts):
            raise RuntimeError("Bitstring length mismatch.")

        rows.append({
            **plan["records"][index],
            "counts": counts,
        })

    if len(rows) != info["end"] - info["start"]:
        raise RuntimeError("Incomplete result.")

    save(output, {
        "job_id": info["job_id"],
        "backend": BACKEND,
        "shots": SHOTS,
        "qpy_sha256": plan["qpy_sha256"],
        "start": info["start"],
        "end": info["end"],
        "records": rows,
    })

    info["status"] = "COLLECTED"
    info["results_path"] = str(output)
    save(JOBS, jobs)

    print("Saved:", output)
    print("Records:", len(rows))
    print("Shots:", len(rows) * SHOTS)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight")

    p = sub.add_parser("submit")
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--confirm-submit", action="store_true")

    p = sub.add_parser("collect")
    p.add_argument("--batch", type=int, required=True)

    args = parser.parse_args()

    if args.command == "preflight":
        preflight()
    elif args.command == "submit":
        submit(args.batch, args.confirm_submit)
    elif args.command == "collect":
        collect(args.batch)


if __name__ == "__main__":
    main()
