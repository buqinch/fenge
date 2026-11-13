#!/usr/bin/env python3
"""Mirror remote experiment text, publishing validated epoch boundaries last.

Only SSH/rsync reads run remotely. Remote process identifiers are provenance;
this program never starts, stops, signals, or resumes a trainer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = frozenset((".json", ".jsonl", ".log", ".md", ".csv", ".txt"))


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_config(path, project=PROJECT):
    """Additional deployment metadata is retained without affecting transfers."""
    path, project = Path(path).resolve(), Path(project).resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema") != "lusnar_remote_text_sync_v1":
        raise ValueError("Expected lusnar_remote_text_sync_v1 configuration")
    host = config.get("host", "fenge103")
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-"):
        raise ValueError("host must be an SSH alias or user@host")
    remote_project = config.get("remote_project", "/home/lzc/workspace/fenge")
    if (not isinstance(remote_project, str) or not PurePosixPath(remote_project).is_absolute()
            or ".." in PurePosixPath(remote_project).parts or "\n" in remote_project):
        raise ValueError("remote_project must be an absolute remote directory")
    interval = config.get("interval_seconds", 60)
    timeout = config.get("timeout_seconds", 120)
    if type(interval) is not int or interval < 1 or type(timeout) is not int or timeout < 1:
        raise ValueError("interval_seconds and timeout_seconds must be positive integers")

    def local_path(raw_path):
        value = Path(raw_path)
        value = value.resolve() if value.is_absolute() else (project / value).resolve()
        if not value.is_relative_to(project / "result"):
            raise ValueError("Local synchronization destinations must be inside project/result")
        return value

    targets = []
    entries = config.get("runs")
    if not isinstance(entries, list) or not entries or not all(isinstance(row, dict) for row in entries):
        raise ValueError("The remote mirror requires a nonempty list of experiment targets")
    names = [row.get("name") for row in entries]
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
           or name == "runner" for name in names):
        raise ValueError("Experiment names must be safe directory names; runner is reserved")
    if len(set(names)) != len(names):
        raise ValueError("Experiment names must be unique")
    for row in [{"name": "runner", **config.get("runner", {})}, *entries]:
        name, relative = row["name"], row.get("remote")
        if (not isinstance(relative, str) or not relative or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts or "\n" in relative):
            raise ValueError("Each remote target must be a relative directory inside remote_project")
        if not isinstance(row.get("local"), str):
            raise ValueError("Each synchronization target needs a local directory")
        targets.append({"name": name, "remote": str(PurePosixPath(remote_project) / relative),
                        "local": local_path(row["local"])})
    if len({target["local"] for target in targets}) != len(targets):
        raise ValueError("Synchronization target directories must be distinct")
    state_path = local_path(config.get("state_path", "result/remote103/sync_state.json"))
    analysis_plan, analysis_output = config.get("analysis_plan"), config.get("analysis_output")
    if bool(analysis_plan) != bool(analysis_output):
        raise ValueError("analysis_plan and analysis_output must be supplied together")
    return {"raw": config, "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "host": host, "remote_project": remote_project.rstrip("/"), "interval_seconds": interval,
            "timeout_seconds": timeout, "targets": targets, "state_path": state_path,
            "analysis_plan": local_path(analysis_plan) if analysis_plan else None,
            "analysis_output": local_path(analysis_output) if analysis_output else None,
            "identity": {"host": host, "remote_project": remote_project.rstrip("/"),
                         "targets": [{**target, "local": str(target["local"])} for target in targets]}}


def invoke(command, timeout):
    return subprocess.run(command, text=True, capture_output=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


def fetch(config, target, stage, run=invoke):
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    probe = run([*ssh, config["host"], "test -d " + shlex.quote(target["remote"])], config["timeout_seconds"])
    if probe.returncode == 1:
        return False
    if probe.returncode:
        raise RuntimeError("Remote directory check failed: " + probe.stderr.strip()[-2000:])
    target["local"].mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-rt", "--checksum", "--safe-links", "--prune-empty-dirs", "--max-size=64m",
               "--timeout=60", "--compare-dest=" + str(target["local"]),
               "-e", shlex.join(ssh)]
    command += ["--include=*/"]
    command += ["--include=*" + suffix for suffix in sorted(TEXT_SUFFIXES)]
    command += ["--exclude=*", config["host"] + ":" + shlex.quote(target["remote"].rstrip("/") + "/"), str(stage) + "/"]
    result = run(command, config["timeout_seconds"])
    if result.returncode:
        raise RuntimeError(f"rsync failed ({result.returncode}): " + result.stderr.strip()[-2000:])
    return True


def read_text_files(stage):
    """Validate every incoming JSON; retain only complete JSONL records."""
    files, documents = {}, {}
    for source in sorted(stage.rglob("*")):
        if source.is_symlink():
            raise ValueError("Remote result symlinks cannot be published")
        if not source.is_file():
            continue
        relative = source.relative_to(stage)
        if source.suffix not in TEXT_SUFFIXES:
            raise ValueError("Non-text extension reached result staging")
        # This also rejects binary payloads mislabeled as textual results.
        text = source.read_text(encoding="utf-8")
        if "\0" in text:
            raise ValueError(f"Binary content in remote text result: {relative}")
        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON number: {value}")
        if source.suffix == ".json":
            documents[relative] = json.loads(text, parse_constant=reject_constant)
        elif source.suffix == ".jsonl":
            lines = text.splitlines(keepends=True)
            if lines and not lines[-1].endswith(("\n", "\r")):
                # An append may be in progress during rsync. Even a parsable
                # unterminated row has not yet committed its newline.
                lines.pop()
                source.write_text("".join(lines), encoding="utf-8")
            for line in lines:
                if line.strip():
                    json.loads(line, parse_constant=reject_constant)
        files[relative] = source
    return files, documents


def boundary(target, files, documents):
    """Return a ready remote summary, or a pending boundary without publishing it."""
    if target["name"] == "runner":
        return None, None
    relative = Path("summary.json")
    if relative not in files:
        return None, None
    summary = documents[relative]
    completed = summary.get("completed_epochs")
    if type(completed) is not int or completed < 0:
        raise ValueError("Remote summary needs a nonnegative completed epoch boundary")
    local_summary = target["local"] / relative
    if local_summary.exists() and json.loads(local_summary.read_text()).get("completed_epochs", -1) > completed:
        raise ValueError("Remote summary moved behind the already published epoch boundary")

    def document(name):
        path = Path(name)
        if path in documents:
            return documents[path]
        local = target["local"] / path
        return json.loads(local.read_text()) if local.is_file() else None

    config = document("run_config.json")
    if config is None:
        return None, "waiting_for_run_config"
    old_config = target["local"] / "run_config.json"
    if old_config.exists() and json.loads(old_config.read_text()) != config:
        raise ValueError("Remote run_config changed after initial publication")
    records = []
    for epoch in range(0 if config.get("epoch_zero_validation") else 1, completed + 1):
        record = document(f"validation_epoch_{epoch:03d}.json")
        if record is None:
            return None, f"waiting_for_validation_epoch_{epoch:03d}"
        if record.get("epoch") != epoch or not isinstance(record.get("validation"), dict):
            raise ValueError("Validation record epoch or structure differs from its filename")
        records.append(record)
    history_path = files.get(Path("history.jsonl"), target["local"] / "history.jsonl")
    if completed and not history_path.is_file():
        return None, "waiting_for_history"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()] if history_path.is_file() else []
    by_epoch = {}
    for record in history:
        epoch = record.get("epoch")
        if type(epoch) is not int or epoch in by_epoch:
            raise ValueError("History contains an invalid or duplicate epoch")
        by_epoch[epoch] = record
    for record in records:
        if record["epoch"] == 0:
            continue
        if record["epoch"] not in by_epoch:
            return None, f"waiting_for_history_epoch_{record['epoch']:03d}"
        if by_epoch[record["epoch"]] != record:
            raise ValueError("History and validation JSON disagree at a completed epoch")
    if records:
        select = lambda record: record["validation"]["miou"] if summary.get("selection_scene") == "custom_validation" else record["validation"]["scenes"]["Moon_5"]["miou"]
        best = max(records, key=select)
        if summary.get("best_epoch") != best["epoch"] or summary.get("best_validation_miou") != select(best):
            raise ValueError("Remote summary best epoch/score differs from its completed validation records")
    return summary, None


def publish(config, target, stage):
    files, documents = read_text_files(stage)
    incoming_config = documents.get(Path("run_config.json"))
    old_config = target["local"] / "run_config.json"
    if incoming_config is not None and old_config.exists() and json.loads(old_config.read_text()) != incoming_config:
        raise ValueError("Remote run_config changed after initial publication")
    summary, pending = boundary(target, files, documents)
    # An old summary remains authoritative until every newly referenced record
    # has arrived. All other validated text can be published for monitoring.
    paths = [p for p in files if p.name != "summary.json"]
    paths.sort(key=lambda p: (p.name != "run_config.json", str(p)))
    if summary is not None or target["name"] == "runner":
        paths += [p for p in files if p.name == "summary.json"]
    write_json(target["local"] / "remote_sync_source.json", {
        "schema": "lusnar_remote_result_source_v1", "source_host": config["host"],
        "remote_directory": target["remote"], "local_directory": str(target["local"]),
        "updated_at": now(), "remote_process_ids_are_local_process_ids": False,
        "process_management": "none; text mirror only", "pending_boundary": pending})
    published = []
    for relative in paths:
        destination = target["local"] / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # os.replace is atomic within the destination filesystem. copy2 keeps
        # timestamps useful for rsync --compare-dest without retaining staging.
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".remote-sync-", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copy2(files[relative], temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        published.append(str(relative))
    local_summary = target["local"] / "summary.json"
    result = {"status": "pending" if pending else "synced", "pending_reason": pending,
              "published_files": len(published), "summary_published": "summary.json" in published,
              "completed_epochs": json.loads(local_summary.read_text())["completed_epochs"] if target["name"] != "runner" and local_summary.exists() else None}
    return result


def sync_once(config, previous=None, run=invoke):
    state = dict(previous or {})
    if state.get("configuration_identity") not in (None, config["identity"]):
        raise ValueError("Existing synchronization state belongs to different remote/local targets")
    state.update(schema="lusnar_remote_text_sync_state_v1", configuration_identity=config["identity"],
                 config_path=config["path"], config_sha256=config["sha256"], source_host=config["host"],
                 started_at=state.get("started_at", now()), updated_at=now(), status="syncing")
    targets_state = state.setdefault("targets", {})
    parent = config["state_path"].parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / ".sync_staging"
    staging.mkdir(exist_ok=True)
    for target in config["targets"]:
        item = dict(targets_state.get(target["name"], {}))
        item.update(source_host=config["host"], remote_directory=target["remote"],
                    local_directory=str(target["local"]), attempted_at=now())
        try:
            with tempfile.TemporaryDirectory(prefix=target["name"] + "-", dir=staging) as temporary:
                stage = Path(temporary)
                if not fetch(config, target, stage, run=run):
                    item.update(status="pending", pending_reason="remote_directory_not_created", error=None)
                else:
                    item.update(publish(config, target, stage), error=None)
                    if item["status"] == "synced":
                        item["last_success_at"] = now()
        except Exception as error:
            item.update(status="error", error=str(error)[-3000:], failed_at=now())
        targets_state[target["name"]] = item
        write_json(config["state_path"], state)
    statuses = {item["status"] for item in targets_state.values()}
    state.update(status="error" if "error" in statuses else "pending" if "pending" in statuses else "synced", updated_at=now())
    if state["status"] == "synced":
        state["last_success_at"] = now()
    state["analysis"] = analyze_if_idle(config)
    if state["analysis"]["status"] == "error":
        state["status"] = "error"
    write_json(config["state_path"], state)
    return state


def analyze_if_idle(config):
    """Continue all-run comparisons after the local queue exits, under its lock."""
    if config["analysis_plan"] is None:
        return {"status": "disabled"}
    output = config["analysis_output"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "busy", "reason": "local_runner_holds_analysis_lock"}
        try:
            if not config["analysis_plan"].is_file():
                return {"status": "pending", "reason": "analysis_plan_not_created"}
            from prior_ablation import analyze
            result = analyze(json.loads(config["analysis_plan"].read_text()), output)
            return {"status": "updated", "at": now(), "all_runs_complete": result.get("complete", False),
                    "selected_run": result.get("selected_run")}
        except Exception as error:
            return {"status": "error", "at": now(), "error": str(error)[-3000:]}
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/remote103.json")
    parser.add_argument("--once", action="store_true", help="One attempt; missing remote directories are pending")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["state_path"].parent.mkdir(parents=True, exist_ok=True)
    with config["state_path"].with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(config["state_path"].read_text()) if config["state_path"].exists() else None
        while True:
            state = sync_once(config, state)
            print(json.dumps({"at": state["updated_at"], "status": state["status"],
                              "targets": {name: {key: item.get(key) for key in ("status", "completed_epochs", "error", "pending_reason")}
                                          for name, item in state["targets"].items()}}, ensure_ascii=False), flush=True)
            if args.once:
                return int(state["status"] == "error")
            time.sleep(config["interval_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())
