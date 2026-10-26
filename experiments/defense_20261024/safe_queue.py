"""Run a finite experiment plan under the current user, yielding to other GPU users.

The supervisor may signal only process groups it created and whose UID, session,
and Linux start time still match. It never changes GPU settings or other jobs.
"""
from __future__ import annotations

import argparse
import atexit
from collections import deque
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time


def process_identity(pid):
    try:
        p = Path('/proc') / str(pid)
        uid = int(next(s for s in (p / 'status').read_text().splitlines()
                       if s.startswith('Uid:')).split()[1])
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return uid, int(fields[2]), int(fields[3]), int(fields[19])
    except (OSError, ValueError, StopIteration):
        return None


def may_signal(pid, expected, current, uid):
    return (expected is not None and current == expected
            and expected[0] == uid and expected[1] == pid and expected[2] == pid)


def signal_job(job, sig):
    proc = job['proc']
    if proc.poll() is not None:
        return False
    now = process_identity(proc.pid)
    if not may_signal(proc.pid, job['identity'], now, os.getuid()):
        raise RuntimeError(f'Refusing to signal an unverified process group {proc.pid}')
    os.killpg(proc.pid, sig)
    return True


def read_health(root, own_pids, cpus):
    h = {'time': time.time(), 'load1': os.getloadavg()[0], 'foreign_gpu_pids': []}
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
    h['available_gib'] = mem['MemAvailable'] / 2**20
    h['swap_used_gib'] = (mem['SwapTotal'] - mem['SwapFree']) / 2**20
    h['disk_free_gib'] = shutil.disk_usage(root).free / 2**30
    temps = []
    for p in Path('/sys/class/hwmon').glob('*/name'):
        try:
            if p.read_text().strip() != 'coretemp':
                continue
            for t in p.parent.glob('temp*_input'):
                v = float(t.read_text()) / 1000
                if 0 < v < 125:
                    temps.append(v)
        except OSError:
            continue
    h['cpu_c'] = max(temps) if temps else None
    frequencies = []
    for cpu in cpus:
        try:
            frequencies.append(int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text()) / 1000)
        except OSError:
            pass
    h['selected_cpu_mhz'] = frequencies
    h['selected_cpu_stat'] = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in {f'cpu{c}' for c in cpus}:
            values = [int(v) for v in fields[1:9]]
            h['selected_cpu_stat'][fields[0]] = {'total': sum(values), 'idle': sum(values[3:5])}
    try:
        pressure = Path('/proc/pressure/cpu').read_text().splitlines()[0].split()
        h['cpu_pressure10'] = float(next(x.split('=')[1] for x in pressure if x.startswith('avg10=')))
    except (OSError, StopIteration):
        h['cpu_pressure10'] = 0.
    try:
        gpu = subprocess.check_output([
            'nvidia-smi', '--id=0',
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'], text=True, timeout=8).strip().split(',')
        h.update(gpu_used_mib=float(gpu[0]), gpu_total_mib=float(gpu[1]),
                 gpu_util=float(gpu[2]), gpu_c=float(gpu[3]), gpu_watts=float(gpu[4]))
        apps = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True, timeout=8)
        h['compute_pids'] = [int(s.strip()) for s in apps.splitlines() if s.strip().isdigit()]
        h['foreign_gpu_pids'] = [p for p in h['compute_pids'] if p not in own_pids]
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        h['gpu_error'] = str(exc)
    return h


def health_reason(h, starting=False):
    if h.get('gpu_error'):
        return 'GPU monitor unavailable'
    if h.get('foreign_gpu_pids'):
        return 'another GPU workload is active'
    if h['disk_free_gib'] < 100:
        return 'less than 100 GiB disk reserve'
    if h['available_gib'] < (32 if starting else 24):
        return 'RAM reserve'
    if h.get('cpu_c') is None:
        return 'CPU temperature monitor unavailable'
    if h['cpu_c'] >= (80 if starting else 94):
        return 'CPU temperature'
    if h['cpu_median_c'] >= (77 if starting else 86):
        return 'sustained CPU temperature'
    if h['gpu_c'] >= (74 if starting else 82):
        return 'GPU temperature'
    if h['load1'] > (20 if starting else 26) or h.get('cpu_pressure10', 0) > (40 if starting else 65):
        return 'shared CPU pressure'
    free = h['gpu_total_mib'] - h['gpu_used_mib']
    if free < (8192 if starting else 3072):
        return 'GPU memory reserve'
    return None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    os.replace(temporary, path)


def dependencies_ready(job, completed, failed, blocked):
    terminal = completed | set(failed) | blocked
    return (all(d in completed for d in job.get('after', []))
            and all(d in terminal for d in job.get('after_any', [])))


def output_complete(root, job):
    if job.get('training'):
        p = root / job['run_dir'] / 'status.json'
        try:
            s = json.loads(p.read_text())
            return s.get('status') == 'complete' and s.get('ood_evaluated', False)
        except (OSError, ValueError):
            return False
    if job.get('success_json'):
        try:
            if json.loads((root / job['success_json']).read_text()).get('passed') is not True:
                return False
        except (OSError, ValueError):
            return False
    return bool(job.get('outputs')) and all((root / p).is_file() and (root / p).stat().st_size > 0
                                           for p in job['outputs'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--deadline', required=True, help='ISO timestamp with UTC offset')
    args = parser.parse_args()
    root = args.root.resolve()
    if root.stat().st_uid != os.getuid() or str(root).startswith('/home/ikovshov/'):
        raise SystemExit('The experiment root must belong to the current user')
    ops = root / 'ops'
    ops.mkdir(exist_ok=True)
    lock = (ops / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads(args.plan.read_text())
    jobs = plan['jobs']
    ids = {j['id'] for j in jobs}
    if len(ids) != len(jobs):
        raise SystemExit('Duplicate job ids')
    for j in jobs:
        if any('/home/ikovshov' in str(x) for x in j['cmd']):
            raise SystemExit('Command refers to the read-only source account')
        for name in [*j.get('outputs', []), *([j['run_dir']] if 'run_dir' in j else [])]:
            if not (root / name).resolve().is_relative_to(root):
                raise SystemExit('Output escapes the owned experiment root')
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    write_json(ops / 'queue_pid.json', {'pid': os.getpid(), 'uid': os.getuid(), 'root': str(root)})
    active, completed, failed = {}, set(), {}
    history = deque(maxlen=4)
    cool_until = 0.
    previous_health = None

    def cleanup_owned_children():
        for a in list(active.values()):
            if a['proc'].poll() is None:
                try:
                    signal_job(a, signal.SIGTERM)
                except (OSError, RuntimeError):
                    continue
        for a in list(active.values()):
            try:
                a['proc'].wait(timeout=120)
            except subprocess.TimeoutExpired:
                signal_job(a, signal.SIGKILL)

    atexit.register(cleanup_owned_children)

    def log(event, **kw):
        row = {'time': datetime.now().astimezone().isoformat(), 'event': event, **kw}
        with (ops / 'queue_events.jsonl').open('a') as f:
            f.write(json.dumps(row, default=str) + '\n')
        print(json.dumps(row, default=str), flush=True)

    log('queue_start', uid=os.getuid(), jobs=len(jobs), deadline=args.deadline)
    while True:
        for jid, a in list(active.items()):
            proc = a['proc']
            rc = proc.poll()
            if rc is not None:
                a['log'].close()
                del active[jid]
                j = a['job']
                if a.get('stop_reason'):
                    log('checkpointed', job=jid, code=rc, reason=a['stop_reason'])
                elif rc == 0 and output_complete(root, j):
                    completed.add(jid)
                    log('complete', job=jid, seconds=round(time.time() - a['started'], 1))
                else:
                    failed[jid] = {'exit_code': rc, 'outputs_complete': output_complete(root, j)}
                    log('failed', job=jid, **failed[jid])
                continue
            if a.get('stop_at') and time.time() - a['stop_at'] > 120:
                signal_job(a, signal.SIGKILL)
                log('force_exit_own_job', job=jid)
        own_pids = {a['proc'].pid for a in active.values()}
        h = read_health(root, own_pids, list(range(16, 24)))
        if h['cpu_c'] is not None:
            history.append(h['cpu_c'])
        h['cpu_median_c'] = statistics.median(history) if history else 999.
        if previous_health:
            utilization = {}
            for cpu, counters in h['selected_cpu_stat'].items():
                old = previous_health['selected_cpu_stat'].get(cpu)
                if old:
                    total = counters['total'] - old['total']
                    utilization[cpu] = round(100 * (1 - (counters['idle'] - old['idle']) / total), 1) if total else 0.
            h['selected_cpu_utilization'] = utilization
        previous_health = h
        with (ops / 'telemetry.jsonl').open('a') as f:
            f.write(json.dumps({**h, 'own_active': {k: v['proc'].pid for k, v in active.items()}}) + '\n')
        reason = 'supervisor stop' if stop else 'deadline' if time.time() >= deadline else health_reason(h)
        for jid, a in active.items():
            if reason and not a.get('stop_reason'):
                signal_job(a, signal.SIGTERM)
                a.update(stop_reason=reason, stop_at=time.time())
                cool_until = time.time() + 45
                log('request_checkpoint', job=jid, reason=reason)
        for j in jobs:
            jid = j['id']
            if jid not in completed and jid not in active and jid not in failed and output_complete(root, j):
                completed.add(jid)
                log('reuse_complete', job=jid)
        blocked = set()
        while True:
            extended = {j['id'] for j in jobs if any(d in set(failed) | blocked for d in j.get('after', []))}
            if extended == blocked:
                break
            blocked = extended
        if not stop and time.time() < deadline and not reason and time.time() >= cool_until:
            try:
                control = json.loads((ops / 'control.json').read_text())
            except (OSError, ValueError):
                control = {'max_parallel': 1}
            maximum = max(0, min(2, int(control.get('max_parallel', 1))))
            for j in jobs:
                jid = j['id']
                if len(active) >= maximum:
                    break
                if jid in active or jid in completed or jid in failed or jid in blocked:
                    continue
                if not dependencies_ready(j, completed, failed, blocked):
                    continue
                if health_reason(h, starting=True):
                    break
                # Both allocations leave all performance cores and eight efficiency cores free.
                occupied = {a['lane'] for a in active.values()}
                lane = next(x for x in [0, 1] if x not in occupied)
                cpus = '16-19' if lane == 0 else '20-23'
                command = list(j['cmd'])
                if j.get('training') and (root / j['run_dir'] / 'last.pt').exists():
                    command += ['--resume', str(root / j['run_dir'] / 'last.pt')]
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                       'VIRTUAL_ENV': str(root / 'env_mentors/.venv'),
                       'PATH': str(root / 'env_mentors/.venv/bin') + os.pathsep + os.environ.get('PATH', '')}
                log_file = (ops / f'{jid}.log').open('a')
                proc = subprocess.Popen(['taskset', '-c', cpus, 'nice', '-n', '10', *command],
                                        cwd=root, env=env, stdout=log_file,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                ident = process_identity(proc.pid)
                if not may_signal(proc.pid, ident, ident, os.getuid()):
                    raise RuntimeError('Could not establish identity of the child process')
                active[jid] = {'proc': proc, 'identity': ident, 'job': j, 'log': log_file,
                               'started': time.time(), 'lane': lane}
                log('start', job=jid, pid=proc.pid, cpus=cpus, resume='--resume' in command)
        pending = [j['id'] for j in jobs if j['id'] not in completed | set(failed) | blocked | set(active)]
        status = {'time': datetime.now().astimezone().isoformat(), 'pid': os.getpid(),
                  'completed': sorted(completed), 'failed': failed, 'blocked': sorted(blocked),
                  'pending': pending, 'active': {k: {'pid': a['proc'].pid, 'seconds': round(time.time() - a['started']),
                                                     'stopping': a.get('stop_reason')} for k, a in active.items()},
                  'health': h, 'waiting_reason': reason or health_reason(h, starting=True)}
        write_json(ops / 'queue_status.json', status)
        if not active and (stop or time.time() >= deadline or not pending):
            write_json(ops / 'queue_summary.json', {**status, 'finished': True})
            log('queue_end', completed=len(completed), failed=failed, blocked=sorted(blocked), pending=pending)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
