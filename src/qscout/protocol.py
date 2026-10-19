"""Guards that make the one-shot test-set evaluation auditable."""

from __future__ import annotations

import subprocess
from pathlib import Path


class ProtocolError(RuntimeError):
    pass


def _git(args: list[str], cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def check_protocol_frozen(
    tag: str, repo: str | Path = ".", paths: tuple[str, ...] = ("docs/PROTOCOL.md", "configs")
) -> str:
    """Require that `tag` exists and that the protocol and configs are identical to that tag
    (committed and unmodified). Returns the tagged commit hash."""
    r = _git(["rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], repo)
    if r.returncode != 0:
        raise ProtocolError(f"git tag '{tag}' not found; freeze the protocol first: git tag {tag}")
    commit = r.stdout.strip()
    d = _git(["diff", "--quiet", tag, "--", *paths], repo)
    if d.returncode != 0:
        raise ProtocolError(f"{', '.join(paths)} differ from tag '{tag}'; the protocol is not frozen")
    return commit


def acquire_final_lock(out_dir: str | Path, commit: str, force: bool = False) -> Path:
    """The test set is evaluated once. A second attempt requires --force and is logged."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".final.lock"
    if lock.exists() and not force:
        raise ProtocolError(f"{lock} exists: the test set was already evaluated ({lock.read_text().strip()})")
    mode = "w" if force else "x"
    with open(lock, mode) as f:
        f.write(f"commit={commit}\n")
    return lock
