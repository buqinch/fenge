"""Wait for completed experiment analysis, commit result records, and push main.

Only the fixed result directories below enter the commit. Training checkpoints,
source files, existing staged work, credentials, and Git's global settings are
never changed by this publisher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[1]
RUN_NAMES = ("dual_region_balanced8000_bs8", "dual_region_balanced8000_bs8_tta")
RESULT_DIRECTORIES = tuple("result/" + name for name in RUN_NAMES) + ("result/analysis", "result/experiment_monitor")
MONITOR_FILES = {"state.json", "launch.json", "monitor.log"}
SMALL_SUFFIXES = {".json", ".jsonl", ".log", ".csv", ".md", ".txt", ".png", ".svg"}
MAX_FILE_BYTES = 10 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def sanitize(value):
    value = re.sub(r"(https?://)[^/@\s]+@", r"\1[redacted]@", str(value))
    value = re.sub(r"(?i)(authorization\s*[:=]\s*)[^\r\n]+", r"\1[redacted]", value)
    value = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)\b", "[redacted]", value)
    return value[:4000]


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(project, arguments):
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith("GIT_TRACE") or key == "GIT_CURL_VERBOSE":
            environment.pop(key)
    environment.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never", GIT_SSL_NO_VERIFY="false",
                       GIT_ASKPASS="/bin/false", SSH_ASKPASS="/bin/false", SSH_ASKPASS_REQUIRE="never",
                       GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=15")
    command = ["git", "-c", "http.sslVerify=true", "-c", "core.hooksPath=/dev/null", *arguments]
    try:
        result = subprocess.run(command, cwd=project, env=environment, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Git command exceeded the 60-second timeout") from error
    if result.returncode:
        raise RuntimeError(sanitize(result.stderr or result.stdout or f"Git exited {result.returncode}"))
    return result.stdout


def target_remote(value):
    value = value.strip().rstrip("/")
    return re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
                        r"buqinch/fenge(?:\.git)?", value, re.IGNORECASE) is not None


def verify_git(project):
    if Path(run_git(project, ["rev-parse", "--show-toplevel"]).strip()).resolve() != project:
        raise RuntimeError("Publisher must run in this project's own Git repository")
    if run_git(project, ["branch", "--show-current"]).strip() != "main":
        raise RuntimeError("Refusing publication from a branch other than main")
    for arguments in (["remote", "get-url", "--all", "origin"],
                      ["remote", "get-url", "--push", "--all", "origin"]):
        remotes = run_git(project, arguments).splitlines()
        if not remotes or not all(target_remote(value) for value in remotes):
            raise RuntimeError("Origin must identify the authorized github.com/buqinch/fenge repository")
    if run_git(project, ["diff", "--cached", "--name-only", "-z"]):
        raise RuntimeError("Existing staged content detected; refusing to mix another change into the results commit")


def verify_artifacts(project, monitor):
    analysis = project / "result/analysis"
    for name in ("report_zh.md", "summary.json", "metrics.csv", "confusion_matrices.csv", "history.csv"):
        path = analysis / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise RuntimeError(f"Final analysis artifact is missing: result/analysis/{name}")
    summary = read_json(analysis / "summary.json")
    if summary.get("state") != "complete" or summary.get("schema") != "lusnar_experiment_analysis_v1":
        raise RuntimeError("Final analysis summary is incomplete")
    expected = {(project / "result" / name).resolve() for name in RUN_NAMES}
    if {Path(row["run_dir"]).resolve() for row in summary.get("runs", [])} != expected:
        raise RuntimeError("Final analysis does not identify exactly the two authorized runs")
    if monitor.get("analysis_output") and Path(monitor["analysis_output"]).resolve() != analysis.resolve():
        raise RuntimeError("Monitor analysis output differs from the fixed publication directory")
    for name in RUN_NAMES:
        run = project / "result" / name
        training = read_json(run / "summary.json")
        if training.get("status") not in ("complete", "early_stopped", "stopped") or training.get("partial") is not False:
            raise RuntimeError(f"Experiment is not fully finalized: {name}")
        for mode, filename in (("none", "test_metrics.json"), ("hflip", "test_hflip_metrics.json")):
            result = read_json(run / filename)
            if (result.get("image_count") != 5004 or result.get("partial") is not False
                    or result.get("tta_mode") != mode or result.get("split") != "test"
                    or result.get("checkpoint_epoch") != training["best_epoch"]):
                raise RuntimeError(f"Final test metrics are incomplete: {name}/{filename}")


def selected_files(project):
    files = []
    for relative in RESULT_DIRECTORIES:
        directory = project / relative
        if directory.is_symlink():
            raise RuntimeError("Publication directory must not be a symlink")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != project):
                continue
            if not path.resolve().is_relative_to(directory.resolve()):
                raise RuntimeError("Publication artifact escaped its permitted directory")
            if relative == "result/experiment_monitor" and path.name not in MONITOR_FILES:
                continue
            if path.suffix.lower() not in SMALL_SUFFIXES or path.stat().st_size > MAX_FILE_BYTES:
                continue
            files.append(path.relative_to(project).as_posix())
    return files


def publish_once(monitor_path, project=None):
    project = (project or PROJECT).resolve()
    publication_path = project / "result/experiment_monitor/publication.json"
    publication_path.parent.mkdir(parents=True, exist_ok=True)
    previous = read_json(publication_path) if publication_path.exists() else {}
    record = {**previous, "updated_at": now(), "monitor_state": str(monitor_path.resolve())}
    committed = previous.get("commit")
    try:
        if not monitor_path.is_file() or read_json(monitor_path).get("status") != "ready_to_publish":
            record.update(status="waiting_for_analysis")
            write_json(publication_path, record)
            return record
        monitor = read_json(monitor_path)
        verify_artifacts(project, monitor)
        verify_git(project)
        files = selected_files(project)
        fingerprints = {name: sha256(project / name) for name in files}
        if committed:
            if not re.fullmatch(r"[0-9a-f]{40,64}", committed):
                raise RuntimeError("Saved publication commit is invalid")
            if run_git(project, ["rev-parse", "HEAD"]).strip() != committed:
                raise RuntimeError("HEAD changed since the results commit; refusing to push unrelated later commits")
            if previous.get("artifacts_sha256") != fingerprints:
                raise RuntimeError("Experiment records changed after the local publication commit")
        else:
            run_git(project, ["add", "--", *files])
            staged = [name for name in run_git(project, ["diff", "--cached", "--name-only", "-z"]).split("\0") if name]
            if set(staged) - set(files):
                raise RuntimeError("Unexpected staged paths appeared; refusing the results commit")
            if staged:
                try:
                    run_git(project, ["commit", "--only", "-m",
                                      "Record completed dual CNN experiments and evaluation analysis", "--", *staged])
                except Exception:
                    run_git(project, ["restore", "--staged", "--", *staged])
                    raise
            committed = run_git(project, ["rev-parse", "HEAD"]).strip()
            record.update(commit=committed, artifacts_sha256=fingerprints, files=files, locally_committed_at=now())
        record.update(status="locally_committed", commit=committed, error=None)
        write_json(publication_path, record)  # Durable before attempting the external write.
        if previous.get("status") != "published":
            run_git(project, ["push", "origin", "main"])
        record.update(status="published", published_at=previous.get("published_at", now()), error=None)
    except Exception as error:
        record.update(status="push_failed" if committed else "publication_failed", error=sanitize(error))
        if committed:
            record["commit"] = committed
    write_json(publication_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor-state", type=Path, default=PROJECT / "result/experiment_monitor/state.json")
    parser.add_argument("--once", action="store_true", help="Return immediately if analysis is not ready")
    args = parser.parse_args()
    directory = PROJECT / "result/experiment_monitor"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "publisher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            record = publish_once(args.monitor_state)
            if record["status"] != "waiting_for_analysis" or args.once:
                print(json.dumps({key: record.get(key) for key in ("status", "commit", "error")}, ensure_ascii=False))
                return 0 if record["status"] in ("waiting_for_analysis", "published") else 1
            time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
