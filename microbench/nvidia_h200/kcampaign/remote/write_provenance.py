"""Record which code snapshot, runner files and plan a kcampaign run used.

Writes runs/<plan>/_provenance.json (one entry appended per start), since the
per-window launch.json only stores runner_relpath.
"""
import datetime, hashlib, json, socket, sys
from pathlib import Path

repo, plan_path, out_dir, gpu = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
note = sys.argv[5] if len(sys.argv) > 5 else 'written at launch'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plans = json.loads(plan_path.read_text())['plans']
runners = sorted({p['runner_relpath'] for p in plans})
entry = {
    'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'note': note, 'host': socket.gethostname(), 'physical_gpu': gpu,
    'repo_snapshot': str(repo), 'plan_file': str(plan_path),
    'plan_sha256': sha(plan_path), 'case_count': len(plans),
    'files_sha256': {r: sha(repo / r) for r in runners + [
        'scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py',
        'dataset_pipeline/kernel_dataset/nvidia_h200_capture.py']},
}
out_dir.mkdir(parents=True, exist_ok=True)
path = out_dir / '_provenance.json'
entries = json.loads(path.read_text()) if path.exists() else []
entries.append(entry)
path.write_text(json.dumps(entries, indent=1))
print(path, len(entries))
