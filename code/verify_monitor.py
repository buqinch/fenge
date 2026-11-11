"""CPU checks for monitor recovery and process identity; never signal real jobs."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import monitor_experiments as monitor
from lunarseg.early_stopping import PatienceTracker


def save(path, value):
    path.write_text(json.dumps(value))


def expect_error(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("Unsafe state was accepted")


def main():
    policy = {"patience": 5, "min_delta": 0.0005, "start_epoch": 4}
    assert PatienceTracker().min_delta == 0.0005
    with tempfile.TemporaryDirectory(prefix="lusnar-monitor-check-") as directory:
        output = Path(directory)
        summary = {"status": "running", "completed_epochs": 8, "best_epoch": 4,
                   "best_validation_miou": 0.8}
        save(output / "summary.json", summary)
        for epoch in range(1, 10):
            save(output / f"validation_epoch_{epoch:03d}.json", {
                "epoch": epoch, "partial": False,
                "validation": {"partial": False, "image_count": 1599,
                               "scenes": {"Moon_5": {"miou": 0.8}}}})
        _, tracker = monitor.completed_tracker(output, policy)
        assert tracker.last_epoch == 8 and tracker.bad_epochs == 4 and not tracker.stopped
        summary["completed_epochs"] = 9
        save(output / "summary.json", summary)
        _, tracker = monitor.completed_tracker(output, policy)
        assert tracker.stopped and tracker.stop_epoch == 9
        print("PASS summary commit boundary excludes future validation JSON")

        launch = {"pid": 987654321, "command": ["python", "train.py"], "gpu_uuid": "GPU-test"}
        identity = {"pid": launch["pid"], "starttime_ticks": 123,
                    "command": launch["command"], "cwd": str(monitor.PROJECT), "pgid": launch["pid"]}
        with patch.object(monitor, "process_alive", side_effect=[True, False]), \
                patch.object(monitor, "process_identity", return_value=identity), \
                patch.object(monitor.os, "killpg") as signal_group:
            monitor.stop_training(launch, output, identity)
            signal_group.assert_called_once_with(launch["pid"], monitor.signal.SIGTERM)
        for changed in ({**identity, "command": ["unrelated"]},
                        {**identity, "starttime_ticks": 124},
                        {**identity, "pgid": 1}):
            with patch.object(monitor, "process_alive", return_value=True), \
                    patch.object(monitor, "process_identity", return_value=changed), \
                    patch.object(monitor.os, "killpg") as signal_group:
                expect_error(lambda: monitor.stop_training(launch, output, identity))
                signal_group.assert_not_called()
        with patch.object(monitor, "process_alive", return_value=True), \
                patch.object(monitor, "process_identity", return_value=None), \
                patch.object(monitor.os, "killpg") as signal_group:
            monitor.stop_training(launch, output, identity)
            signal_group.assert_not_called()
        with patch.object(monitor, "process_alive", return_value=True), \
                patch.object(monitor, "process_identity", return_value=identity), \
                patch.object(monitor.os, "killpg", side_effect=ProcessLookupError):
            monitor.stop_training(launch, output, identity)
        print("PASS SIGTERM only for exact command/cwd/group/starttime; disappearing PID is harmless")

        save(output / "launch.json", launch)
        (output / "best.pt").write_bytes(b"current-checkpoint")
        current_sha = monitor.digest(output / "best.pt")
        terminal = {**summary, "status": "complete"}
        binding = {"name": "test", "output": str(output), "primary_tta": "none",
                   "launch_sha256": monitor.digest(output / "launch.json")}
        run = {"name": "test", "output": str(output), "primary_tta": "none"}
        with patch.object(monitor, "capture_training_identity", return_value=None), \
                patch.object(monitor, "completed_tracker", return_value=(terminal, tracker)), \
                patch.object(monitor, "launch_evaluation") as launch_eval:
            expect_error(lambda: monitor.advance_run(run, policy, {**binding, "output": "/another/run"}))
            expect_error(lambda: monitor.advance_run(run, policy, {**binding, "launch_sha256": "old-launch"}))
            expect_error(lambda: monitor.advance_run(run, policy, {**binding, "best_checkpoint_sha256": "old-checkpoint"}))
            launch_eval.assert_not_called()
        monitor._HASH_CACHE.clear()
        with patch.object(monitor, "digest", wraps=monitor.digest) as hasher:
            monitor.checkpoint_digest(output / "best.pt")
            monitor.checkpoint_digest(output / "best.pt")
            assert hasher.call_count == 1
            monitor._HASH_CACHE.clear()  # A restarted monitor must verify once again.
            monitor.checkpoint_digest(output / "best.pt")
            assert hasher.call_count == 2
        print("PASS changed output/launch/checkpoint rejected; digest cached only within this process")

        intent_path, registration_path = monitor.evaluation_paths(output, "none")
        intent = {"launch_id": "unique", "tta": "none", "output": str(output / "test_metrics.json"),
                  "log": str(output / "test.log"), "started_at": "now", "command": ["python", "evaluate.py"],
                  "checkpoint_sha256": current_sha, "checkpoint_epoch": 4,
                  "gpu_uuid": "GPU-test", "registration": str(registration_path)}
        save(intent_path, intent)
        with patch.object(monitor.os, "execvpe") as execute:
            monitor.evaluation_worker(intent_path)
            assert registration_path.is_file()
            registration = json.loads(registration_path.read_text())
            assert registration["pid"] > 0 and registration["starttime_ticks"] > 0
            execute.assert_called_once()
        with patch.object(monitor, "process_identity", return_value=None):
            recovered = monitor.recovered_evaluation(output, "none", current_sha, 4)
            assert recovered["launch_id"] == "unique"
            expect_error(lambda: monitor.recovered_evaluation(output, "none", "changed", 4))
        with patch.object(monitor.fcntl, "flock", side_effect=BlockingIOError), \
                patch.object(monitor.os, "execvpe") as execute:
            assert monitor.evaluation_worker(intent_path) == 0
            execute.assert_not_called()
        print("PASS worker registers before exec, restart recovers identity, flock blocks duplicate execution")

        child = MagicMock()
        child.poll.return_value = 0
        with patch.dict(monitor._CHILDREN, {54321: child}, clear=True):
            assert not monitor.process_alive(54321)
            child.poll.assert_called_once()
            assert 54321 not in monitor._CHILDREN
        print("PASS completed Popen child polled/reaped")
    print("ALL_MONITOR_CHECKS_PASSED")


if __name__ == "__main__":
    main()
