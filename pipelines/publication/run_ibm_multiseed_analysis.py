"""Evaluate seed 101 IBM hardware measurements using the locked protocol."""

import csv
import json
from collections import defaultdict
from pathlib import Path

from pipelines.publication.run_iris_validation import (
    make_iris_dataset,
    evaluate_fixed_parity,
    evaluate_representation,
    decoder_templates,
    LEARNED_REPRESENTATIONS,
)

ROOT = Path("results/publication/ibm_hardware_validation")
SEED = None
OUTPUT = None
SHOTS = 1024

def main(seed):
    global SEED, OUTPUT
    SEED = seed
    OUTPUT = ROOT / f"seed{seed}_hardware_metrics.csv"

    plan = json.loads((ROOT / "resumable_plan.json").read_text())
    original_jobs = json.loads((ROOT / "jobs.json").read_text())
    accelerated_jobs = json.loads(
        (ROOT / "accelerated/jobs.json").read_text()
    )

    indexed_rows = {}

    sources = [
        (
            ROOT / f"batch_{batch:03d}_counts.json",
            original_jobs[str(batch)],
        )
        for batch in range(5)
    ]

    for batch, info in sorted(
        accelerated_jobs.items(),
        key=lambda item: int(item[0]),
    ):
        if info["status"] == "COLLECTED":
            sources.append((
                ROOT / "accelerated" /
                f"batch_{int(batch):03d}_counts.json",
                info,
            ))

    for path, info in sources:
        if info["status"] != "COLLECTED":
            raise RuntimeError(f"Uncollected source: {path}")

        payload = json.loads(path.read_text())

        if payload["job_id"] != info["job_id"]:
            raise RuntimeError("Job ID mismatch.")

        if payload["qpy_sha256"] != plan["qpy_sha256"]:
            raise RuntimeError("Frozen QPY checksum mismatch.")

        for offset, row in enumerate(payload["records"]):
            index = info["start"] + offset

            if index in indexed_rows:
                raise RuntimeError("Duplicate circuit index.")

            expected = plan["records"][index]

            for key in ("seed", "split", "sample", "basis"):
                if row[key] != expected[key]:
                    raise RuntimeError("Record mapping mismatch.")

            if sum(row["counts"].values()) != SHOTS:
                raise RuntimeError("Shot count mismatch.")

            if any(len(k) != 4 for k in row["counts"]):
                raise RuntimeError("Invalid bitstring.")

            indexed_rows[index] = row

    grouped = defaultdict(dict)

    for row in indexed_rows.values():
        if row["seed"] != SEED:
            continue

        key = (row["split"], row["sample"])
        basis = row["basis"]

        if basis in grouped[key]:
            raise RuntimeError("Duplicate sample/basis.")

        grouped[key][basis] = row["counts"]

    expected_sizes = {
        "train": 60,
        "validation": 20,
        "test": 20,
    }

    records = {}

    for split, size in expected_sizes.items():
        records[split] = []

        for sample in range(size):
            key = (split, sample)

            if set(grouped[key]) != {"X", "Y", "Z"}:
                raise RuntimeError(f"Incomplete measurement: {key}")

            records[split].append(grouped[key])

    assert len(grouped) == 100

    dataset = make_iris_dataset(SEED)

    results = []

    parity = evaluate_fixed_parity(
        records["test"],
        dataset.y_test,
    )

    results.append({
        "seed": SEED,
        "backend": "ibm_marrakesh",
        "representation": "R0_parity",
        "decoder": "fixed_threshold",
        "feature_dimension": 1,
        **parity,
    })

    for decoder_name, decoder in decoder_templates().items():
        for representation in LEARNED_REPRESENTATIONS:
            metrics = evaluate_representation(
                representation=representation,
                decoder_template=decoder,
                train_records=records["train"],
                validation_records=records["validation"],
                test_records=records["test"],
                y_train=dataset.y_train,
                y_validation=dataset.y_validation,
                y_test=dataset.y_test,
            )

            results.append({
                "seed": SEED,
                "backend": "ibm_marrakesh",
                "representation": representation.value,
                "decoder": decoder_name,
                **metrics,
            })

    if OUTPUT.exists():
        raise RuntimeError(
            "Output already exists. Refusing to overwrite."
        )

    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "backend",
                "representation",
                "decoder",
                "feature_dimension",
                "accuracy",
                "macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nIBM HARDWARE CLASSIFICATION RESULTS")
    print("=" * 75)

    for row in results:
        print(
            f"{row['representation']:20s} "
            f"{row['decoder']:22s} "
            f"Accuracy={row['accuracy']:.3f} "
            f"Macro-F1={row['macro_f1']:.3f}"
        )

    print("\nSaved:", OUTPUT)
    print("Models evaluated:", len(results))
    print("No hardware jobs submitted.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=[101, 202, 303, 404, 505],
    )
    args = parser.parse_args()
    main(args.seed)
