"""Build a local, reproducible source bundle for the Modal Host candidate.

Only committed Midlayer code, one public model geometry file and a shallow
vLLM v0.30.0 sparse checkout are included. No model weights or dataset enter
the bundle. The cloud runner verifies the vLLM source identity again.
"""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT.parent / "upstream/vllm/v0.30.0"
OUTPUT = Path("/tmp/midlayer-h200-modal-scheduler-bundle.tar.gz")
REPO_PATHS = (
    "src",
    "scripts/measure_vllm_scheduler_service.py",
    "configs/model/qwen3-4b.json",
    "microbench/nvidia_h200/runtime/launch/torch_eager_graph_runner.py",
)


def _run(argv: list[str], *, cwd: Path) -> str:
    return subprocess.run(argv, cwd=cwd, check=True, text=True,
                          capture_output=True).stdout.strip()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if _run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT):
        raise RuntimeError("bundle requires a clean Midlayer worktree")
    repo_commit = _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="midlayer-h200-scheduler-pack-",
                                     dir="/tmp") as temporary:
        stage = Path(temporary)
        source = stage / "vllm_source"
        repo = stage / "midlayer_repo"
        _run(["git", "clone", "--depth", "1", "--single-branch", "--branch",
              "v0.30.0", "--no-checkout", UPSTREAM.as_uri(), str(source)],
             cwd=stage)
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=source)
        _run(["git", "sparse-checkout", "set", "vllm"], cwd=source)
        _run(["git", "checkout", "v0.30.0"], cwd=source)
        source_commit = _run(["git", "rev-parse", "HEAD"], cwd=source)
        source_tag = _run(["git", "describe", "--tags", "--exact-match", "HEAD"],
                          cwd=source)
        if (source_tag != "v0.30.0" or
                _run(["git", "status", "--porcelain", "--untracked-files=all"],
                     cwd=source)):
            raise RuntimeError("vLLM source is not a clean v0.30.0 checkout")
        repo.mkdir()
        archive = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD", *REPO_PATHS],
            cwd=ROOT, check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as reader:
            reader.extractall(repo, filter="data")
        source_hashes = {}
        for relative in REPO_PATHS[1:]:
            path = repo / relative
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "repo_commit": repo_commit,
            "source_commit": source_commit,
            "source_tag": source_tag,
            "repo_paths": REPO_PATHS,
            "source_hashes": source_hashes,
            "contents": "committed Midlayer src and independent probes; public vLLM source; no weights or datasets",
        }
        (stage / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        with tarfile.open(OUTPUT, mode="w:gz") as writer:
            writer.add(stage / "bundle_manifest.json", arcname="bundle_manifest.json")
            writer.add(source, arcname="vllm_source")
            writer.add(repo, arcname="midlayer_repo")
    print(json.dumps({"output": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      "repo_commit": repo_commit, "source_commit": source_commit},
                     sort_keys=True))


if __name__ == "__main__":
    main()
