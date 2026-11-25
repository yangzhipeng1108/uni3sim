"""Loader for the ProtoBridge upstream surrogate linked as a git submodule.

The upstream source is no longer vendored into ``src/kernel/predictors/
protobridge/``.  It lives in the ``third_party/protobridge`` submodule pinned
to the commit recorded in ``PROVENANCE.md``.  This loader resolves the pinned
file by content, so a stale or missing checkout fails loud instead of
silently importing an unverified upstream revision.
"""

from pathlib import Path
import hashlib
import importlib.util
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SUBMODULE_RELATIVE_PATH = "third_party/protobridge"
PROVENANCE_PATH = (
    Path(__file__).resolve().parent / "PROVENANCE.md")

_COMMIT_RE = re.compile(r"^- Upstream commit: `([0-9a-f]{40})`$", re.M)
_SOURCE_SHA_RE = re.compile(r"^- Source SHA256: `([0-9a-f]{64})`$", re.M)
_LICENSE_SHA_RE = re.compile(
    r"^- License: GPL-2.0-only.*SHA256 `([0-9a-f]{64})`", re.M)
_UPSTREAM_PATH_RE = re.compile(
    r"^- Upstream path: `([^`]+)`$", re.M)


class _ProvenanceParseError(RuntimeError):
    pass


def _parse_provenance() -> dict:
    """Extract the pinned upstream identity from the PROVENANCE owner file."""
    text = PROVENANCE_PATH.read_text()
    commit = _COMMIT_RE.search(text)
    source_sha = _SOURCE_SHA_RE.search(text)
    license_sha = _LICENSE_SHA_RE.search(text)
    upstream_path = _UPSTREAM_PATH_RE.search(text)
    if not (commit and source_sha and license_sha and upstream_path):
        raise _ProvenanceParseError(
            "PROVENANCE.md must pin exactly one upstream commit, source "
            "SHA256, license SHA256, and upstream path")
    return {
        "commit": commit.group(1),
        "source_sha256": source_sha.group(1),
        "license_sha256": license_sha.group(1),
        "upstream_path": upstream_path.group(1),
    }


def pinned_upstream_identity() -> dict:
    """Return the upstream commit, source, and license digests pinned by
    ``PROVENANCE.md``."""
    return _parse_provenance()


def _fail_closed(reason: str) -> None:
    raise RuntimeError(
        f"ProtoBridge upstream surrogate is unavailable: {reason}. "
        "Run `git submodule update --init --recursive` from the repository "
        "root and verify the pinned commit matches PROVENANCE.md "
        "(expected upstream commit "
        f"{_parse_provenance()['commit']}).")


def _pinned_source_path() -> Path:
    pinned = _parse_provenance()
    return (
        REPOSITORY_ROOT / SUBMODULE_RELATIVE_PATH / pinned["upstream_path"])


def upstream_file_sha256() -> str:
    """Return the SHA256 of the linked upstream surrogate source file."""
    pinned = _parse_provenance()
    source_path = (
        REPOSITORY_ROOT / SUBMODULE_RELATIVE_PATH / pinned["upstream_path"])
    if not source_path.is_file():
        _fail_closed(
            f"expected file {source_path} does not exist; the submodule "
            "checkout is missing or checked out at a different revision")
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != pinned["source_sha256"]:
        _fail_closed(
            f"{source_path} has SHA256 {digest}, but PROVENANCE.md pins "
            f"{pinned['source_sha256']}")
    return digest


def submodule_checked_out_commit() -> str:
    """Return the commit currently checked out in the submodule."""
    gitdir_pointer = (
        REPOSITORY_ROOT / SUBMODULE_RELATIVE_PATH / ".git")
    if not gitdir_pointer.exists():
        _fail_closed(
            "third_party/protobridge is not initialized as a git submodule")
    module_git_dir = gitdir_pointer.read_text().strip()
    if not module_git_dir.startswith("gitdir: "):
        _fail_closed(
            "third_party/protobridge/.git does not point to a gitdir")
    # The gitdir pointer is relative to the submodule worktree; resolve it
    # against that directory before anchoring at the repository root.
    pointer_path = (
        REPOSITORY_ROOT / SUBMODULE_RELATIVE_PATH
        / module_git_dir[len("gitdir: "):].strip())
    if not pointer_path.is_dir():
        _fail_closed(
            f"cannot resolve the submodule gitdir {pointer_path}")
    head_path = pointer_path / "HEAD"
    if not head_path.is_file():
        _fail_closed("cannot resolve the submodule HEAD file")
    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref_path = REPOSITORY_ROOT / ".git" / head[len("ref: "):].strip()
        if not ref_path.is_file():
            _fail_closed(f"cannot resolve the submodule ref {head[5:]}")
        return ref_path.read_text().strip()
    return head


def load_upstream_module():
    """Import the pinned upstream ``whitebox_surrogate_v2`` module.

    The module is loaded from its canonical upstream path so tracebacks and
    ``inspect``-based tooling report the real upstream file, while the digest
    check above guarantees the exact source revision is what gets imported.
    """
    source_path = _pinned_source_path()
    upstream_file_sha256()
    spec = importlib.util.spec_from_file_location(
        "third_party.protobridge.surrogate.theory.whitebox_surrogate_v2",
        source_path,
    )
    if spec is None or spec.loader is None:
        _fail_closed(f"cannot create an import spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_upstream_license() -> str:
    """Verify the linked upstream LICENSE digest matches PROVENANCE.md."""
    pinned = _parse_provenance()
    license_path = REPOSITORY_ROOT / SUBMODULE_RELATIVE_PATH / "LICENSE"
    if not license_path.is_file():
        _fail_closed(f"expected file {license_path} does not exist")
    digest = hashlib.sha256(license_path.read_bytes()).hexdigest()
    if digest != pinned["license_sha256"]:
        _fail_closed(
            f"{license_path} has SHA256 {digest}, but PROVENANCE.md pins "
            f"{pinned['license_sha256']}")
    return digest


def load_upstream_module_and_verify_license():
    """Import the upstream module after verifying both source and license."""
    verify_upstream_license()
    return load_upstream_module()
