"""Monitor completed validation JSON, stop on patience, then evaluate and analyze.

The trainer is unchanged. summary.json is written after last.pt/best.pt, so its
completed_epochs field is the commit boundary for observing an epoch safely.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from lunarseg.early_stopping import PatienceTracker

PROJECT = Path(__file__).resolve().parents[1]
_CHILDREN = {}
_HASH_CACHE = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_alive(pid):
    child = _CHILDREN.get(int(pid))
    if child is not None and child.poll() is not None:
        _CHILDREN.pop(int(pid), None)
        return False
    try:
        state = (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1].split()[0]
        if state in ("Z", "X"):
            try:
                os.waitpid(int(pid), os.WNOHANG)
            except (ChildProcessError, ProcessLookupError):
                pass
        return state not in ("Z", "X")
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_identity(pid):
    proc = Path("/proc") / str(pid)
    try:
        fields = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] in ("Z", "X"):
            return None
        return {"pid": int(pid), "starttime_ticks": int(fields[19]),
                "command": [item.decode() for item in (proc / "cmdline").read_bytes().split(b"\0") if item],
                "cwd": str((proc / "cwd").resolve(strict=True)), "pgid": os.getpgid(int(pid))}
    except (FileNotFoundError, ProcessLookupError):
        return None


def identity_alive(identity):
    if not identity or not process_alive(identity["pid"]):
        return False
    current = process_identity(identity["pid"])
    return current is not None and current["starttime_ticks"] == identity["starttime_ticks"]


def capture_training_identity(launch, previous=None):
    current = process_identity(launch["pid"])
    if current is None:
        return previous
    if previous and current["starttime_ticks"] != previous["starttime_ticks"]:
        return previous  # The original process has exited; never adopt a reused PID.
    if current["command"] != launch["command"] or current["cwd"] != str(PROJECT) or current["pgid"] != int(launch["pid"]):
        raise RuntimeError("Trainer process identity differs from its recorded launch")
    if launch.get("starttime_ticks") is not None and launch["starttime_ticks"] != current["starttime_ticks"]:
        raise RuntimeError("Trainer PID was reused after the recorded launch")
    return current


def stop_training(launch, output, identity=None):
    """Stop only the dedicated process group created for this exact run."""
    pid = int(launch["pid"])
    if not process_alive(pid):
        return
    current = process_identity(pid)
    if current is None:
        return
    if identity and current["starttime_ticks"] != identity["starttime_ticks"]:
        raise RuntimeError(f"Refusing to signal reused PID {pid}")
    if current["command"] != launch["command"] or current["cwd"] != str(PROJECT):
        raise RuntimeError(f"Refusing to signal PID {pid}: process identity differs from {output}")
    if current["pgid"] != pid:
        raise RuntimeError(f"Refusing to signal PID {pid}: no dedicated training process group")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(100):
        if not process_alive(pid):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Training PID {pid} did not terminate after SIGTERM")


def completed_tracker(output, policy):
    summary = read_json(output / "summary.json")
    completed = summary["completed_epochs"]
    records = []
    custom = summary.get("selection_scene") == "custom_validation"
    custom_identity = None
    for epoch in range(1, completed + 1):
        record = read_json(output / f"validation_epoch_{epoch:03d}.json")
        validation = record["validation"]
        identity = validation.get("data_identity", {})
        if custom or identity.get("custom_split") is True:
            scenes = validation.get("scenes", {})
            allowed = {"Moon_1", "Moon_3", "Moon_5", "Moon_6", "Moon_7", "Moon_8"}
            if (not custom or summary.get("partial") is not False
                    or summary.get("official_test_used_for_selection") is not False
                    or record["epoch"] != epoch or record.get("partial") is not False
                    or record.get("selection_scene") != "custom_validation"
                    or validation.get("partial") is not False or validation.get("image_count") != 1601
                    or identity.get("custom_split") is not True or identity.get("split") != "val"
                    or identity.get("protocol") != "random_original_source_family_4_to_1"
                    or identity.get("image_count") != 1601 or identity.get("validation_image_count") != 1601
                    or identity.get("partial") is not False or len(scenes) < 2 or not set(scenes).issubset(allowed)
                    or set(identity.get("validation_scenes", [])) != set(scenes)
                    or any(type(row.get("image_count")) is not int or row["image_count"] < 1 for row in scenes.values())
                    or sum(row["image_count"] for row in scenes.values()) != 1601):
                raise ValueError(f"Early stopping requires complete 1601-image custom validation: {output}, epoch {epoch}")
            if custom_identity is not None and identity != custom_identity:
                raise ValueError("Custom validation identity changed within completed history")
            custom_identity = identity
            if record.get("selection_score") != validation.get("miou"):
                raise ValueError("Custom checkpoint selection score differs from pooled validation mIoU")
        elif (record["epoch"] != epoch or record.get("partial") or validation.get("partial")
              or validation["image_count"] != 1599 or set(validation["scenes"]) != {"Moon_5"}):
            raise ValueError(f"Early stopping requires complete Moon_5 validation: {output}, epoch {epoch}")
        records.append(record)
    tracker_policy = dict(policy)
    if custom:
        if tracker_policy.get("metric", "custom_validation_miou") != "custom_validation_miou":
            raise ValueError("Custom validation cannot use the Moon_5 patience metric")
        tracker_policy["metric"] = "custom_validation_miou"
    tracker = PatienceTracker(**tracker_policy)
    tracker.replay(records, completed_epochs=completed)
    return summary, tracker


def analysis_command(config):
    """Build the final analysis command without launching it."""
    comparison = config.get("comparison", "tta")
    runs = config["runs"]
    if comparison == "random-split":
        if len(runs) != 1 or config.get("reference_tta") not in ("none", "hflip") or not config.get("reference_run"):
            raise ValueError("Random-split monitoring needs one new run, reference_run and reference_tta")
        reference = resolve(config["reference_run"])
        current = resolve(runs[0]["output"])
        if reference == current:
            raise ValueError("Random-split reference and new experiment must be different runs")
        return [sys.executable, "-B", str(PROJECT / "code/analyze_experiments.py"),
                "--comparison", "random-split", "--run-off", str(reference), "--run-on", str(current),
                "--reference-tta", config["reference_tta"], "--output", str(resolve(config["analysis_output"]))]
    if comparison not in ("tta", "hard-pixels") or len(runs) != 2:
        raise ValueError("Scene-protocol monitoring needs two runs and comparison tta or hard-pixels")
    return [sys.executable, "-B", str(PROJECT / "code/analyze_experiments.py"),
            "--run-off", str(resolve(runs[0]["output"])), "--run-on", str(resolve(runs[1]["output"])),
            "--output", str(resolve(config["analysis_output"])), "--comparison", comparison]


def checkpoint_digest(path):
    stat = path.stat()
    fingerprint = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    key = str(path.resolve())
    cached = _HASH_CACHE.get(key)
    if cached is None or cached[0] != fingerprint:
        cached = (fingerprint, digest(path))
        _HASH_CACHE[key] = cached
    return cached[1], fingerprint


def validate_test(path, checkpoint_sha, epoch, tta):
    result = read_json(path)
    expected = {"Moon_2": 1597, "Moon_4": 1320, "Moon_9": 2087}
    if (result.get("partial") or result.get("checkpoint_partial_training")
            or result.get("image_count") != 5004 or result.get("split") != "test"
            or result.get("checkpoint_sha256") != checkpoint_sha
            or result.get("checkpoint_epoch") != epoch or result.get("tta_mode") != tta):
        raise ValueError(f"Invalid or stale complete test result: {path}")
    if {scene: row["image_count"] for scene, row in result["scenes"].items()} != expected:
        raise ValueError(f"Incomplete test scene counts: {path}")
    return result


def evaluation_paths(output, tta):
    return output / f"evaluation_{tta}_intent.json", output / f"evaluation_{tta}_launch.json"


def evaluation_worker(intent_path):
    """Register before exec; an inherited flock prevents duplicate evaluation."""
    intent = read_json(intent_path)
    lock = intent_path.with_suffix(".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0  # A survivor from the previous monitor owns this exact job.
    identity = process_identity(os.getpid())
    registration = {**intent, "pid": os.getpid(), "starttime_ticks": identity["starttime_ticks"],
                    "worker_command": identity["command"], "cwd": str(PROJECT), "registered_at": now()}
    write_json(Path(intent["registration"]), registration)
    os.set_inheritable(lock.fileno(), True)
    os.execvpe(intent["command"][0], intent["command"], os.environ.copy())


def recovered_evaluation(output, tta, checkpoint_sha, epoch):
    intent_path, registration_path = evaluation_paths(output, tta)
    if not registration_path.exists():
        return None
    registration = read_json(registration_path)
    if (registration["checkpoint_sha256"] != checkpoint_sha or registration["checkpoint_epoch"] != epoch
            or registration["tta"] != tta or registration["output"] != str(output / ("test_hflip_metrics.json" if tta == "hflip" else "test_metrics.json"))):
        raise RuntimeError("Evaluation registration belongs to another checkpoint or run")
    if not intent_path.exists() or read_json(intent_path)["launch_id"] != registration["launch_id"]:
        raise RuntimeError("Evaluation registration/intent identity mismatch")
    current = process_identity(registration["pid"])
    if current and current["starttime_ticks"] == registration["starttime_ticks"]:
        if current["command"] not in (registration["command"], registration["worker_command"]) or current["cwd"] != str(PROJECT):
            raise RuntimeError("Active evaluation process identity differs from its registration")
    return registration


def launch_evaluation(output, launch, tta, checkpoint_sha, epoch):
    target = output / ("test_hflip_metrics.json" if tta == "hflip" else "test_metrics.json")
    log_path = output / ("test_hflip.log" if tta == "hflip" else "test.log")
    command = [sys.executable, "-B", "-u", str(PROJECT / "code/evaluate.py"),
               "--checkpoint", str(output / "best.pt"), "--split", "test",
               "--batch-size", "2", "--workers", "4", "--device", "cuda:0",
               "--amp", "bf16", "--tta", tta, "--output", str(target)]
    intent_path, registration_path = evaluation_paths(output, tta)
    # Persist the intent before any child can exist. A restarted monitor can
    # recover a self-registered child even if its own state write never happened.
    if intent_path.exists():
        intent = read_json(intent_path)
        if (intent["checkpoint_sha256"] != checkpoint_sha or intent["checkpoint_epoch"] != epoch
                or intent["command"] != command or intent["gpu_uuid"] != launch["gpu_uuid"]):
            raise RuntimeError("Existing evaluation intent differs from this run/checkpoint")
    else:
        intent = {"launch_id": uuid.uuid4().hex, "tta": tta, "output": str(target), "log": str(log_path),
                  "started_at": now(), "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": epoch,
                  "command": command, "gpu_uuid": launch["gpu_uuid"], "registration": str(registration_path)}
        write_json(intent_path, intent)
    recovered = recovered_evaluation(output, tta, checkpoint_sha, epoch)
    if recovered:
        return recovered
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=launch["gpu_uuid"], OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    with log_path.open("ab", buffering=0) as log:
        worker_command = [sys.executable, "-B", str(Path(__file__).resolve()), "--evaluation-worker", str(intent_path)]
        process = subprocess.Popen(worker_command, cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    _CHILDREN[process.pid] = process
    identity = process_identity(process.pid)
    return {**intent, "pid": process.pid, "starttime_ticks": identity["starttime_ticks"] if identity else None,
            "worker_command": worker_command, "registration_pending": True, "spawned_at": time.time()}


def advance_run(run, policy, previous):
    output = resolve(run["output"])
    launch = read_json(output / "launch.json")
    launch_sha = digest(output / "launch.json")
    binding = {"name": run["name"], "output": str(output), "launch_sha256": launch_sha,
               "primary_tta": run["primary_tta"]}
    if previous and any(previous.get(key) != value for key, value in binding.items()):
        raise RuntimeError("Monitor state belongs to another config/output/training launch")
    training_identity = capture_training_identity(launch, previous.get("training_process_identity"))
    if not (output / "summary.json").exists():
        if not identity_alive(training_identity):
            raise RuntimeError(f"Trainer exited before completing its first epoch: {output}")
        state = {**previous, **binding, "training_process_identity": training_identity,
                 "completed_epochs": 0, "phase": "training_first_epoch"}
        write_json(output / "early_stopping_monitor.json", state)
        return state
    summary, tracker = completed_tracker(output, policy)
    state = {**previous, **binding, "training_process_identity": training_identity,
             "completed_epochs": summary["completed_epochs"], "best_epoch": summary["best_epoch"],
             "best_validation_miou": summary["best_validation_miou"], "early_stopping": tracker.state_dict()}
    if previous.get("completed_epochs") != summary["completed_epochs"]:
        print(f"{run['name']}: epoch={summary['completed_epochs']} best={summary['best_validation_miou']:.6%} "
              f"no_significant_growth={tracker.bad_epochs}/{tracker.patience}", flush=True)
    if tracker.stopped and summary["status"] == "running":
        # Recheck the boundary after reading the per-epoch files.
        if read_json(output / "summary.json")["completed_epochs"] != summary["completed_epochs"]:
            return state
        for filename in ("last.pt", "best.pt"):
            if not (output / filename).is_file():
                raise FileNotFoundError(output / filename)
        stop_training(launch, output, training_identity)
        summary, tracker = completed_tracker(output, policy)
        summary.update(status="early_stopped", stopped_at=now(),
                       stop_reason=tracker.stop_reason, early_stopping=tracker.state_dict(),
                       stop_mechanism="external_completed_validation_json_monitor")
        write_json(output / "summary.json", summary)
        state.update(completed_epochs=summary["completed_epochs"], best_epoch=summary["best_epoch"],
                     best_validation_miou=summary["best_validation_miou"], early_stopping=tracker.state_dict())
        print(f"{run['name']}: early stopped after epoch {summary['completed_epochs']}; best checkpoint preserved", flush=True)
    if summary["status"] == "running":
        if not identity_alive(training_identity):
            raise RuntimeError(f"Trainer exited unexpectedly before a completed/early_stopped summary: {output}")
        state["phase"] = "training"
    elif summary["status"] in ("complete", "early_stopped", "stopped"):
        if identity_alive(training_identity):
            state["phase"] = "waiting_for_trainer_exit"
            return state
        current_sha, fingerprint = checkpoint_digest(output / "best.pt")
        if state.get("best_checkpoint_sha256") not in (None, current_sha):
            raise RuntimeError("Best checkpoint changed after finalization; refusing cached evaluations")
        if state.get("best_checkpoint_epoch") not in (None, summary["best_epoch"]):
            raise RuntimeError("Best checkpoint epoch changed after finalization")
        if "best_checkpoint_sha256" not in state:
            state["best_checkpoint_sha256"] = current_sha
            state["stopped_at"] = summary.get("stopped_at", now())
        state.update(best_checkpoint_epoch=summary["best_epoch"], best_checkpoint_fingerprint=fingerprint)
        checkpoint_sha = state["best_checkpoint_sha256"]
        evaluations = state.setdefault("evaluations", {})
        active = state.get("active_evaluation")
        if active:
            active = recovered_evaluation(output, active["tta"], checkpoint_sha, summary["best_epoch"]) or active
        if active:
            if process_alive(active["pid"]):
                current = process_identity(active["pid"])
                if current and active.get("starttime_ticks") is not None and current["starttime_ticks"] != active["starttime_ticks"]:
                    raise RuntimeError("Evaluation PID was reused; refusing to wait on another process")
                state["active_evaluation"] = active
                state["phase"] = "evaluating_" + active["tta"]
                write_json(output / "early_stopping_monitor.json", state)
                return state
            if active.get("registration_pending") and time.time() - active.get("spawned_at", 0) < 15:
                state["phase"] = "waiting_for_evaluation_registration"
                return state
            result = validate_test(Path(active["output"]), checkpoint_sha, summary["best_epoch"], active["tta"])
            evaluations[active["tta"]] = {**active, "completed_at": now(),
                                          "weighted_score": result["weighted_score"]}
            state.pop("active_evaluation")
            print(f"{run['name']}: full test {active['tta']} S={result['weighted_score']:.6%}", flush=True)
        modes = [run["primary_tta"], "hflip" if run["primary_tta"] == "none" else "none"]
        for tta in modes:
            if tta in evaluations:
                validate_test(Path(evaluations[tta]["output"]), checkpoint_sha, summary["best_epoch"], tta)
                continue
            target = output / ("test_hflip_metrics.json" if tta == "hflip" else "test_metrics.json")
            if target.is_file():
                result = validate_test(target, checkpoint_sha, summary["best_epoch"], tta)
                evaluations[tta] = {"output": str(target), "weighted_score": result["weighted_score"], "reused": True}
                continue
            state["active_evaluation"] = launch_evaluation(output, launch, tta, checkpoint_sha, summary["best_epoch"])
            state["phase"] = "evaluating_" + tta
            print(f"{run['name']}: evaluating best epoch {summary['best_epoch']} on complete Moon_2/4/9 ({tta})", flush=True)
            break
        else:
            state["phase"] = "evaluated"
    else:
        raise RuntimeError(f"Unexpected trainer status {summary['status']!r}: {output}")
    write_json(output / "early_stopping_monitor.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config/early_stopping_monitor.json")
    parser.add_argument("--evaluation-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inspect", action="store_true", help="Compute JSON decisions without writing, signaling, or launching jobs")
    args = parser.parse_args()
    if args.evaluation_worker:
        evaluation_worker(args.evaluation_worker)
        return
    config = read_json(args.config)
    if args.inspect:
        for run in config["runs"]:
            run_output = resolve(run["output"])
            if not (run_output / "summary.json").exists():
                launch_path = run_output / "launch.json"
                launch = read_json(launch_path) if launch_path.exists() else None
                identity = capture_training_identity(launch) if launch else None
                print(json.dumps({"run": run["name"], "completed_epochs": 0,
                                  "status": "training_first_epoch" if identity_alive(identity) else "not_running",
                                  "policy": config["policy"]}, ensure_ascii=False))
                continue
            summary, tracker = completed_tracker(run_output, config["policy"])
            print(json.dumps({"run": run["name"], "status": summary["status"], **tracker.state_dict()}, ensure_ascii=False))
        return
    final_analysis_command = analysis_command(config)
    output = resolve(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = output / "state.json"
    state = read_json(path) if path.exists() else {"started_at": now(), "runs": {}}
    config_sha = digest(args.config)
    if path.exists() and state.get("config_sha256") != config_sha:
        raise RuntimeError("Monitor configuration changed; refusing to reuse another configuration's state")
    if set(state["runs"]) - {run["name"] for run in config["runs"]}:
        raise RuntimeError("Monitor state contains runs absent from this configuration")
    state.update(monitor_pid=os.getpid(), policy=config["policy"], config_sha256=config_sha)
    try:
        while True:
            for run in config["runs"]:
                state["runs"][run["name"]] = advance_run(run, config["policy"], state["runs"].get(run["name"], {}))
                state.update(status="monitoring", updated_at=now())
                write_json(path, state)
            if all(run["phase"] == "evaluated" for run in state["runs"].values()):
                subprocess.run(final_analysis_command, cwd=PROJECT, check=True)
                state.update(status="ready_to_publish", analysis_output=str(resolve(config["analysis_output"])), completed_at=now())
                write_json(path, state)
                print("Configured experiments evaluated; final analysis ready for Git publication.", flush=True)
                return
            time.sleep(config.get("poll_seconds", 5))
    except BaseException as error:
        state.update(status="monitor_error", error=f"{type(error).__name__}: {error}", updated_at=now())
        write_json(path, state)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
