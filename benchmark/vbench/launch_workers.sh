#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Usage: bash $0 TASK MODEL PRECISION ROOT OMNI_SOURCE GPU_LIST [I2V_IMAGES]"
    echo "GPU_LIST is comma-separated physical GPU indices (e.g. 0,1). Uses PYTHON or python3."
    echo "Prepare once, generate disjoint shards, wait for every worker, then verify. No evaluation."
    exit 0
fi
if [[ $# != 6 && $# != 7 ]]; then echo "Expected TASK MODEL PRECISION ROOT OMNI_SOURCE GPU_LIST [I2V_IMAGES]" >&2; exit 2; fi
py=${PYTHON:-python3}
runner=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/benchmark.py
export DIFFUSION_ATTENTION_BACKEND=TRTLLM_ATTN
export PYTHONUNBUFFERED=1
# exec forwards signals to the supervisor. Every managed child has its own session;
# cleanup signals only process groups created by this invocation, never other runs.
exec "$py" - "$runner" "$@" <<'PY'
import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

parser = argparse.ArgumentParser(description='Fixed-protocol worker supervisor (no evaluation)')
parser.add_argument('runner', type=Path)
parser.add_argument('task', choices=('t2v', 'i2v'))
parser.add_argument('model', type=Path)
parser.add_argument('precision', choices=('bf16', 'nvfp4', 'mxfp8', 'nvfp4_e5m3'))
parser.add_argument('root', type=Path)
parser.add_argument('omni_source', type=Path)
parser.add_argument('gpu_list')
parser.add_argument('images', type=Path, nargs='?')
args = parser.parse_args()
tokens = args.gpu_list.split(',')
if any(not t.isascii() or not t.isdecimal() for t in tokens):
    parser.error('GPU_LIST must contain nonempty comma-separated nonnegative GPU indices')
gpus = [str(int(t)) for t in tokens]
if len(set(gpus)) != len(gpus):
    parser.error('Duplicate GPU indices are not allowed')
if (args.task == 'i2v') != (args.images is not None):
    parser.error('I2V requires I2V_IMAGES; T2V must not receive it')
for path in (args.model, args.omni_source):
    if not path.expanduser().is_dir():
        parser.error(f'Directory missing: {path}')
root = args.root.expanduser().resolve()
model = args.model.expanduser().resolve()
source = args.omni_source.expanduser().resolve()
spec = importlib.util.spec_from_file_location('_worker_protocol', args.runner)
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
logs = root / 'logs'
logs.mkdir(parents=True, exist_ok=True)
lock = (logs / 'launcher.lock').open('a')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    parser.exit(1, 'Another launcher owns this run root\n')
run_id = uuid.uuid4().hex
report_path = logs / f'workers-{run_id}.json'
report = dict(task=args.task, precision=args.precision, gpu_list=gpus,
              status='running', stages=[], workers=[])
processes = []
spawning = False
pending_signal = None

class Interrupted(BaseException):
    def __init__(self, number):
        self.number = number

def interrupt(number, frame):
    global pending_signal
    if spawning:
        # Defer the exception until the new child is registered for cleanup.
        pending_signal = number
        return
    raise Interrupted(number)

for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(number, interrupt)

def persist():
    protocol.atomic_json(report_path, report)

def start(role, command, gpu, shard=None):
    global spawning, pending_signal
    log_path = logs / f'{run_id}-{role}.log'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               DIFFUSION_ATTENTION_BACKEND='TRTLLM_ATTN', PYTHONUNBUFFERED='1')
    record = dict(role=role, gpu=gpu, shard_index=shard, command=command,
                  log=str(log_path), pid=None, returncode=None)
    (report['workers'] if shard is not None else report['stages']).append(record)
    # Do not block OS signals here: a blocked mask would be inherited by workers.
    spawning = True
    try:
        with log_path.open('wb') as output:
            child = subprocess.Popen(command, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((child, record))
        record['pid'] = child.pid
    finally:
        spawning = False
        if pending_signal is not None:
            number, pending_signal = pending_signal, None
            raise Interrupted(number)
    persist()
    return child, record

def wait(pair):
    child, record = pair
    record['returncode'] = child.wait()
    persist()
    return record['returncode']

def cleanup():
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, signal.SIG_IGN)
    for child, record in processes:
        # Groups are private sessions, including any inference subprocesses.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for child, record in processes:
        try:
            record['returncode'] = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            record['returncode'] = child.wait()

code = 1
try:
    persist()
    print(f'Logs and worker exit records: {report_path}', flush=True)
    base = [sys.executable, str(args.runner)]
    prepare = base + ['prepare', '--task', args.task, '--root', str(root)]
    if args.images is not None:
        prepare += ['--i2v-images', str(args.images.expanduser().resolve())]
    code = wait(start('prepare', prepare, gpus[0]))
    if code == 0:
        workers = []
        for index, gpu in enumerate(gpus):
            command = base + ['generate', '--task', args.task, '--root', str(root),
                              '--model', str(model), '--precision', args.precision,
                              '--omni-source', str(source), '--shard-index', str(index),
                              '--num-shards', str(len(gpus))]
            workers.append(start(f'worker-{index}', command, gpu, index))
        # Do not short circuit: reap every worker even if an earlier shard failed.
        exits = [wait(pair) for pair in workers]
        code = 1 if any(exits) else 0
        if code == 0:
            code = wait(start('verify', base + ['verify', '--task', args.task,
                                               '--root', str(root)], gpus[0]))
    report['status'] = 'succeeded' if code == 0 else 'failed'
except Interrupted as exc:
    code = 128 + exc.number
    report.update(status='interrupted', signal=exc.number)
except Exception as exc:
    code = 1
    report.update(status='failed', error=str(exc))
    print(f'Launcher failed: {exc}', file=sys.stderr)
finally:
    cleanup()
    report['returncode'] = code if 0 <= code <= 255 else 1
    persist()
    lock.close()
sys.exit(report['returncode'])
PY