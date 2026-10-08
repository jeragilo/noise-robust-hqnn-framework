"""Resumable IBM Quantum validation for the frozen Beyond Parity Iris protocol.

Run from repository root with: python -m pipelines.publication.run_ibm_resumable plan
Subcommands: plan, submit, collect. Submission requires --confirm-submit.
Never stores credentials. Never submits automatically during plan or collect.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from qiskit import qpy
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from pipelines.publication.circuits import build_measurement_circuit, initialize_weights
from pipelines.publication.run_iris_validation import make_iris_dataset

ROOT = Path('results/publication/ibm_hardware_validation')
INSTANCE = 'hqnn-thesis-instance'
BACKEND = 'ibm_marrakesh'
SEEDS = (101, 202, 303, 404, 505)
BASES = ('X', 'Y', 'Z')
SHOTS = 1024
BATCH_SIZE = 15

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True))
    temp.replace(path)

def load_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def manifest_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def build_plan():
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=42)
    records, circuits = [], []
    for seed in SEEDS:
        data = make_iris_dataset(seed)
        weights = initialize_weights(4, seed)
        for split, samples in (('train', data.X_train), ('validation', data.X_validation), ('test', data.X_test)):
            for sample_idx, x in enumerate(samples):
                for basis in BASES:
                    qc = build_measurement_circuit(x=x, weights=weights, num_qubits=4, basis=basis)
                    tqc = pm.run(qc)
                    records.append(dict(seed=seed, split=split, sample=sample_idx, basis=basis,
                                        depth=tqc.depth(), operations=dict(tqc.count_ops())))
                    circuits.append(tqc)
    assert len(circuits) == 1500
    ROOT.mkdir(parents=True, exist_ok=True)
    qpy_path = ROOT / 'frozen_circuits.qpy'
    if (ROOT / 'jobs.json').exists() or qpy_path.exists():
        raise RuntimeError('Plan already exists. Refusing to overwrite frozen circuits or job state.')
    with qpy_path.open('wb') as f:
        qpy.dump(circuits, f)
    digest = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    plan = dict(backend=BACKEND, instance=INSTANCE, shots=SHOTS, batch_size=BATCH_SIZE,
                circuit_count=len(circuits), qpy_sha256=digest, records=records)
    atomic_json(ROOT / 'resumable_plan.json', plan)
    print('Saved frozen QPY and plan. Circuits:', len(circuits), 'QPY SHA256:', digest)
    print('No hardware jobs submitted.')

def verified_plan():
    plan = load_json(ROOT / 'resumable_plan.json', None)
    if plan is None:
        raise RuntimeError('Run plan first.')
    path = ROOT / 'frozen_circuits.qpy'
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan['qpy_sha256']:
        raise RuntimeError('Frozen circuit file changed; refusing to proceed.')
    return plan, path

def submit(batch, confirm):
    plan, path = verified_plan()
    if not confirm:
        print('Submission disabled. Re-run with --confirm-submit after checking QPU allocation.')
        return
    start = batch * plan['batch_size']
    end = min(start + plan['batch_size'], plan['circuit_count'])
    if start >= end or batch < 0:
        raise ValueError('Invalid batch index.')
    jobs_path = ROOT / 'jobs.json'
    jobs = load_json(jobs_path, {})
    key = str(batch)
    if key in jobs:
        raise RuntimeError(f'Batch {batch} already recorded: {jobs[key]}. Collect it instead.')
    with path.open('rb') as f:
        circuits = qpy.load(f)
    service = QiskitRuntimeService(instance=INSTANCE)
    backend = service.backend(BACKEND)
    sampler = SamplerV2(mode=backend)
    # Persist an intent before submission so an interrupted call cannot silently be repeated.
    jobs[key] = dict(status='SUBMISSION_ATTEMPTED', start=start, end=end,
                     created_at=datetime.now(timezone.utc).isoformat())
    atomic_json(jobs_path, jobs)
    print(f'Submitting batch {batch}, circuits [{start}, {end}), shots={SHOTS}', flush=True)
    try:
        job = sampler.run(circuits[start:end], shots=SHOTS)
    except Exception as exc:
        jobs[key]['status'] = 'SUBMISSION_UNCERTAIN'
        jobs[key]['error'] = str(exc)
        atomic_json(jobs_path, jobs)
        raise RuntimeError('Submission uncertain: inspect IBM Workloads before retrying.') from exc
    jobs[key].update(status='SUBMITTED', job_id=job.job_id())
    atomic_json(jobs_path, jobs)
    print('IBM Job ID:', job.job_id(), 'Saved:', jobs_path)

def collect(batch):
    plan, _ = verified_plan()
    jobs_path = ROOT / 'jobs.json'
    jobs = load_json(jobs_path, {})
    key = str(batch)
    if key not in jobs or not jobs[key].get('job_id'):
        raise RuntimeError('No saved job ID. Resolve submission state in IBM Workloads first.')
    item = jobs[key]
    service = QiskitRuntimeService(instance=INSTANCE)
    job = service.job(item['job_id'])
    print('Job:', item['job_id'], 'status:', job.status())
    if str(job.status()).upper() not in ('DONE', 'COMPLETED'):
        print('Not completed; nothing collected.')
        return
    result = job.result()
    rows = []
    for offset, pub in enumerate(result):
        counts = pub.data.meas.get_counts()
        if sum(counts.values()) != SHOTS:
            raise RuntimeError('Unexpected shot count.')
        rows.append(dict(**plan['records'][item['start'] + offset], counts=counts))
    if len(rows) != item['end'] - item['start']:
        raise RuntimeError('Incomplete batch; refusing to mark complete.')
    output = ROOT / f'batch_{batch:03d}_counts.json'
    if output.exists():
        raise RuntimeError('Results file already exists; refusing to overwrite.')
    atomic_json(output, dict(job_id=item['job_id'], backend=BACKEND, shots=SHOTS,
                             qpy_sha256=plan['qpy_sha256'], records=rows))
    item['status'] = 'COLLECTED'
    item['results_path'] = str(output)
    atomic_json(jobs_path, jobs)
    print('Saved', output)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    p = sub.add_parser('submit')
    p.add_argument('--batch', type=int, default=0)
    p.add_argument('--confirm-submit', action='store_true')
    p = sub.add_parser('collect')
    p.add_argument('--batch', type=int, default=0)
    args = parser.parse_args()
    if args.command == 'plan': build_plan()
    elif args.command == 'submit': submit(args.batch, args.confirm_submit)
    else: collect(args.batch)

if __name__ == '__main__':
    main()
