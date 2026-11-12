"""Run the fixed prior ablations, monitor validation JSON, and compare results."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from monitor_experiments import (completed_tracker, digest, identity_alive, now,
                                process_identity, resolve, stop_training, write_json)
from lunarseg.metrics import ConfusionMatrix

PROJECT = Path(__file__).resolve().parents[1]
TERMINAL = {"complete", "early_stopped"}


def read(path):
    return json.loads(Path(path).read_text())


def gpu_inventory():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    return [dict(index=int(i), uuid=u.strip(), free_mib=int(m))
            for i, u, m in (line.split(",") for line in text.splitlines())]


def validated_matrix(metrics, image_count):
    raw = np.asarray(metrics["confusion_matrix_gt_rows_pred_cols"])
    if raw.shape != (5, 5) or raw.dtype.kind not in "iu" or (raw < 0).any():
        raise ValueError("Confusion matrix must contain nonnegative integer 5x5 pixel counts")
    matrix = raw.astype(np.int64)
    if int(matrix.sum()) != image_count * 1024 * 1024:
        raise ValueError("Confusion matrix does not cover every full-resolution validation pixel")
    counter = ConfusionMatrix()
    counter.matrix = matrix
    recomputed = counter.compute()
    if abs(recomputed["miou"] - metrics["miou"]) > 1e-12:
        raise ValueError("Stored validation mIoU differs from its confusion matrix")
    for field in ("class_iou", "class_precision", "class_recall"):
        if recomputed[field] != metrics[field]:
            raise ValueError(f"Stored {field} differs from its confusion matrix")
    return matrix


def training_settings(plan, name, config):
    """Allow batch changes only through explicit, effective-batch-8 overrides."""
    baseline = (plan.get("microbatch", 4), plan.get("accumulation", 2), plan.get("effective_batch", 8))
    if any(type(value) is not int for value in baseline) or baseline != (4, 2, 8):
        raise ValueError("The common plan baseline must remain batch_size 4, accumulation 2, effective batch 8")
    overrides = plan.get("per_run_training_overrides", {})
    names = {run["name"] for run in plan["runs"]}
    allowed = {"batch_size", "accumulation", "gradient_checkpointing"}
    if not isinstance(overrides, dict) or not set(overrides).issubset(names):
        raise ValueError("Training overrides must name existing plan runs")
    for run_name, values in overrides.items():
        if not isinstance(values, dict) or not values or not set(values).issubset(allowed):
            raise ValueError(f"Unsupported training override for {run_name}")
        for field in ("batch_size", "accumulation"):
            value = values.get(field, 4 if field == "batch_size" else 2)
            if type(value) is not int or value < 1:
                raise ValueError(f"Invalid {field} override for {run_name}")
        if values.get("batch_size", 4) * values.get("accumulation", 2) != 8:
            raise ValueError(f"Training override must preserve effective batch 8: {run_name}")
        if "gradient_checkpointing" in values and type(values["gradient_checkpointing"]) is not bool:
            raise ValueError(f"Invalid gradient_checkpointing override for {run_name}")
    declared = overrides.get(name, {})
    settings = {"batch_size": declared.get("batch_size", 4),
                "accumulation": declared.get("accumulation", 2)}
    for field, expected in settings.items():
        if type(config.get(field)) is not int or config[field] != expected:
            raise ValueError(f"Actual {field} differs from the explicit plan for {name}: expected {expected}")
    checkpointing = config.get("gradient_checkpointing", False)
    if type(checkpointing) is not bool or ("gradient_checkpointing" in declared
                                         and checkpointing != declared["gradient_checkpointing"]):
        raise ValueError(f"Actual gradient_checkpointing differs from the explicit plan for {name}")
    return {**settings, "effective_batch": settings["batch_size"] * settings["accumulation"],
            "gradient_checkpointing": checkpointing, "explicit_training_overrides": dict(declared)}


def analyze(plan, output):
    """Use only complete validation records; never rank models on test results."""
    rows, identities, initial_hashes, histories, configurations = [], [], [], {}, {}
    for run in plan["runs"]:
        folder = resolve(run["output"])
        if not (folder / "summary.json").exists():
            continue
        summary = read(folder / "summary.json")
        config = read(folder / "run_config.json")
        settings = training_settings(plan, run["name"], config)
        if (config.get("initialization_mode") != "original_pretrained" or not config.get("pretrained")
                or config.get("initial_checkpoint") is not None or config.get("convnext_checkpoint") is not None):
            raise ValueError("This ablation must start from original pretrained backbones")
        if config.get("epoch_zero_validation"):
            raise ValueError("This from-pretraining plan does not select an epoch-zero segmentation model")
        if summary["completed_epochs"] < 1:
            continue
        history = [read(folder / f"validation_epoch_{epoch:03d}.json")
                   for epoch in range(1, summary["completed_epochs"] + 1)]
        for record in history:
            metrics = record["validation"]
            if metrics["partial"] or metrics["image_count"] != 1601:
                raise ValueError("Ablation requires complete fixed validation")
            matrix = validated_matrix(metrics, 1601)
            expected_counts = {"Moon_1": 217, "Moon_3": 253, "Moon_5": 324, "Moon_6": 216, "Moon_7": 334, "Moon_8": 257}
            if {key: value["image_count"] for key, value in metrics["scenes"].items()} != expected_counts:
                raise ValueError("Validation scene image counts differ from the fixed 1601-image split")
            scene_matrices = [validated_matrix(value, expected_counts[key]) for key, value in metrics["scenes"].items()]
            if not np.array_equal(np.stack(scene_matrices).sum(0), matrix):
                raise ValueError("Per-scene confusion matrices do not sum to the pooled matrix")
        best = max(history, key=lambda r: r["validation"]["miou"])
        if best["epoch"] != summary["best_epoch"]:
            raise ValueError("Best epoch must be first strictly highest validation mIoU")
        metrics = best["validation"]
        cm = np.asarray(metrics["confusion_matrix_gt_rows_pred_cols"], dtype=np.int64)
        assigned_hosts = [assignment.get("host") for assignment in plan.get("execution_assignment", {}).values()
                          if run["name"] in assignment.get("managed_runs", [])]
        if len(assigned_hosts) > 1:
            raise ValueError(f"Run assigned to more than one execution host: {run['name']}")
        directions = {}
        for truth, predicted in ((2, 0), (0, 2), (2, 3), (3, 1), (1, 3)):
            directions[f"{truth}->{predicted}"] = {
                "pixels": int(cm[truth, predicted]),
                "gt_conditional_rate": float(cm[truth, predicted] / cm[truth].sum()) if cm[truth].sum() else None}
        row = {"name": run["name"], "control": run.get("control"), "status": summary["status"],
               **settings,
               "execution_host": assigned_hosts[0] if assigned_hosts else None,
               "completed_epochs": summary["completed_epochs"], "best_epoch": best["epoch"],
               "best_validation_miou": metrics["miou"], "class_iou": metrics["class_iou"],
               "class_precision": metrics["class_precision"], "class_recall": metrics["class_recall"],
               "directions": directions, "global_update": history[-1]["global_update"],
               "checkpoint": str(folder / "best.pt"), "best_record": str(folder / f"validation_epoch_{best['epoch']:03d}.json"),
               "scenes": {key: value["miou"] for key, value in metrics["scenes"].items()},
               "training_seconds": sum(r["seconds"] for r in history)}
        identities.append(config["data_identity"])
        initial_hashes.append(config.get("initial_base_model_sha256", config["initial_model_sha256"]))
        histories[run["name"]] = history
        configurations[run["name"]] = config
        rows.append(row)
    if identities and any(value != identities[0] for value in identities):
        raise ValueError("Ablation train/validation identities differ")
    if initial_hashes and len(set(initial_hashes)) != 1:
        raise ValueError("Initial common backbone/fusion/decoder tensors differ")
    if initial_hashes and initial_hashes[0] != plan["expected_common_initial_tensor_sha256"]:
        raise ValueError("Common initialization differs from the original-pretrained reference")
    fields = ("epochs", "freeze_epochs", "head_lr", "backbone_lr", "prior_lr",
              "prior_seed", "prior_warmup_updates", "weight_decay", "gradient_clip", "seed", "horizontal_flip",
              "workers", "amp", "num_points", "val_batch_size", "val_tta", "val_tta_sizes")
    configs = list(configurations.values())
    for config in configs[1:]:
        for field in fields:
            if config.get(field) != configs[0].get(field):
                raise ValueError(f"Shared training/validation setting differs: {field}")
    lookup = {row["name"]: row for row in rows}
    for row in rows:
        control = lookup.get(row["control"])
        row["delta_vs_control_pp"] = ((row["best_validation_miou"] - control["best_validation_miou"]) * 100
                                      if control else None)
        row["comparison_budget_equal"] = bool(control and row["global_update"] == control["global_update"])
        row["batch_layout_differs_from_control"] = bool(control and
            (row["batch_size"], row["accumulation"]) != (control["batch_size"], control["accumulation"]))
        row["comparison_caveats"] = ([
            "Different actual microbatch and accumulation: MobileNet BatchNorm statistics and update counts, "
            "worker-assigned horizontal flips, random point sampling and numerical training trajectories may differ. "
            "The score difference does not isolate the module effect."
        ] if row["batch_layout_differs_from_control"] else [])
        if control:
            common_epoch = min(row["completed_epochs"], control["completed_epochs"])
            first = max(histories[row["name"]][:common_epoch], key=lambda r: r["validation"]["miou"])
            second = max(histories[control["name"]][:common_epoch], key=lambda r: r["validation"]["miou"])
            row["matched_budget"] = {"through_epoch": common_epoch, "optimizer_updates": common_epoch * 1000,
                                     "best_miou": first["validation"]["miou"],
                                     "control_best_miou": second["validation"]["miou"],
                                     "delta_pp": 100 * (first["validation"]["miou"] - second["validation"]["miou"])}
            audit = {"through_epoch": common_epoch, "sample_order_equal": True,
                     "horizontal_flip_mismatch_epochs": [], "sample_and_horizontal_flip_mismatch_epochs": [],
                     "per_epoch": [], "batch_layout_differs": row["batch_layout_differs_from_control"]}
            for a, b in zip(histories[row["name"]][:common_epoch], histories[control["name"]][:common_epoch]):
                equal = {}
                for field in ("sample_order_sha256", "horizontal_flip_sha256", "sample_and_horizontal_flip_sha256"):
                    if not a.get(field) or not b.get(field):
                        raise ValueError(f"Missing training stream audit for {row['name']}: {field}")
                    equal[field] = a[field] == b[field]
                    if not equal[field]:
                        if field == "sample_order_sha256" or not row["batch_layout_differs_from_control"]:
                            raise ValueError(f"Matched training stream differs for {row['name']}: {field}")
                        audit[field.removesuffix("_sha256") + "_mismatch_epochs"].append(a["epoch"])
                audit["per_epoch"].append({"epoch": a["epoch"], **equal})
            audit["horizontal_flip_equal"] = not audit["horizontal_flip_mismatch_epochs"]
            audit["sample_and_horizontal_flip_equal"] = not audit["sample_and_horizontal_flip_mismatch_epochs"]
            row["training_stream_audit"] = audit
    complete = len(rows) == len(plan["runs"]) and all(row["status"] in TERMINAL for row in rows)
    winner = max(rows, key=lambda r: r["best_validation_miou"]) if complete else None
    result = {"updated_at": now(), "complete": complete, "validation_only_selection": True,
              "initialization": "original_pretrained_backbones_and_identical_random_main_head",
              "common_initial_tensor_sha256": initial_hashes[0] if initial_hashes else None,
              "per_run_training_overrides": plan.get("per_run_training_overrides", {}),
              "runs": rows, "selected_run": winner["name"] if winner else None,
              "interpretation": "Independent single-seed experiments. Shared effective batch 8, maximum update budget and patience policy; actual early-stop epochs may differ. Explicitly overridden microbatches can change MobileNet BatchNorm, worker horizontal flips, random point sampling and numerical trajectories; their differences do not isolate a module effect. No combined model has been tested."}
    write_json(output / "comparison.json", result)
    with (output / "comparison.csv").open("w", newline="") as stream:
        fields = ["name", "control", "execution_host", "batch_size", "accumulation", "effective_batch", "gradient_checkpointing",
                  "batch_layout_differs_from_control", "status", "completed_epochs", "best_epoch", "global_update",
                  "best_validation_miou", "delta_vs_control_pp"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 亲和、周围关系与高分辨率细节独立对照", "",
             "所有组从原始 DINOv3 ConvNeXt-Tiny＋ImageNet MobileNetV3-Large 开始，融合和区域头同种子重新初始化；没有加载第29轮分割权重。",
             "固定8000张训练（4000原图＋2000坑增强＋2000山岩增强），同一1601张验证，1024整图、有效batch 8，验证单视图。默认microbatch 4×累积2；显式覆盖见下表。", "",
             "|实验|对照|主机|实际batch×累积|重算|状态|完成轮数|最佳轮次|验证mIoU|相对对照百分点|", "|---|---|---|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        delta = "—" if row["delta_vs_control_pp"] is None else f"{row['delta_vs_control_pp']:+.4f}"
        lines.append(f"|{row['name']}|{row['control'] or '—'}|{row['execution_host'] or '—'}|{row['batch_size']}×{row['accumulation']}|{row['gradient_checkpointing']}|{row['status']}|{row['completed_epochs']}|{row['best_epoch']}|{row['best_validation_miou']:.4%}|{delta}|")
    lines += ["", "亲和、中心—周围关系和细节注入分别与单视图control比较。几何一致性已停用，历史结论保留在清理记录中。",
              "最多30轮并使用相同早停规则；若实际更新数不同，不把最佳分差解释为固定训练预算下的纯模块因果收益。单种子结果尚不能证明稳定提升。",
              "完整类别IoU、精确率、召回率、六场景和混淆方向见[comparison.json](comparison.json)。"]
    overridden = [row for row in rows if row["batch_layout_differs_from_control"]]
    if overridden:
        lines += ["", "**批大小差异限制：** " + "、".join(row["name"] for row in overridden) +
                  " 按显式计划使用不同的实际batch/累积组合。即使有效batch同为8，MobileNet BatchNorm的统计和更新次数、worker分配造成的水平翻转、随机监督点与数值轨迹仍可能不同；相对4×2对照的分差不能只归因于新增模块。",
                  "样本顺序hash仍逐轮强制一致；翻转与样本＋翻转hash逐轮核对，只有显式异batch比较允许报告不一致，具体轮次保存在training_stream_audit。"]
    if not complete:
        lines += ["", "**实验尚未全部结束，表中为阶段性结果，不据此选择最终方案。**"]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n")
    return result


def run(plan_path, output, devices, max_parallel, minimum_free_mib, managed_runs=None):
    plan = read(plan_path)
    if plan.get("status") != "ready" or not plan.get("source_sha256"):
        raise ValueError("Only a ready plan with nonempty frozen source hashes can launch training")
    if not devices or max_parallel < 1 or minimum_free_mib < 1:
        raise ValueError("Provide GPU devices and positive concurrency/memory limits")
    names = [entry["name"] for entry in plan["runs"]]
    if len(names) != len(set(names)):
        raise ValueError("Plan run names must be unique")
    if managed_runs is not None and (not isinstance(managed_runs, (list, tuple))
            or not managed_runs or len(managed_runs) != len(set(managed_runs))
            or not set(managed_runs).issubset(names)):
        raise ValueError("Managed runs must be a nonempty unique subset of plan run names")
    managed = [name for name in names if managed_runs is None or name in managed_runs]
    execution_host = socket.gethostname()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / "state.json"
    state = read(state_path) if state_path.exists() else {"plan_sha256": digest(plan_path), "runs": {}, "status": "running"}
    if state["plan_sha256"] != digest(plan_path):
        raise ValueError("Runner state belongs to a different immutable plan")
    if state.get("managed_runs", managed) != managed:
        raise ValueError("Runner state belongs to a different managed run assignment")
    if state.get("execution_host", execution_host) != execution_host:
        raise ValueError("Runner state belongs to a different execution host; do not adopt its process IDs")
    state.update(managed_runs=managed, execution_host=execution_host,
                 status_scope="managed_runs", status="running")
    children = {}
    for entry in plan["runs"]:
        item = state["runs"].setdefault(entry["name"], {"phase": "queued"})
        item["managed"] = entry["name"] in managed
        if item["managed"]:
            if item.get("execution_host", execution_host) != execution_host:
                raise ValueError(f"Refusing to manage another host's run: {entry['name']}")
            item["execution_host"] = execution_host
    while True:
        for entry in plan["runs"]:
            item = state["runs"][entry["name"]]
            folder = resolve(entry["output"])
            if entry["name"] not in managed:
                # These JSON files may be synchronized from another machine.
                # Never inspect or signal its PID, or write into its run folder.
                item["phase"] = "external"
                if (folder / "summary.json").exists():
                    summary, _ = completed_tracker(folder, plan["early_stopping"])
                    item.update(observed_status=summary["status"],
                                observed_terminal=summary["status"] in TERMINAL,
                                completed_epochs=summary["completed_epochs"],
                                best_epoch=summary["best_epoch"],
                                best_validation_miou=summary["best_validation_miou"])
                else:
                    item.update(observed_status="awaiting_summary", observed_terminal=False)
                continue
            if item["phase"] != "running":
                continue
            child = children.get(entry["name"])
            if child is not None:
                child.poll()
            alive = identity_alive(item["identity"])
            if (folder / "summary.json").exists():
                summary, tracker = completed_tracker(folder, plan["early_stopping"])
                if item.get("completed_epochs") != summary["completed_epochs"]:
                    print(f"{entry['name']}: epoch {summary['completed_epochs']}, best {summary['best_validation_miou']:.6%}", flush=True)
                item.update(completed_epochs=summary["completed_epochs"], best_epoch=summary["best_epoch"],
                            best_validation_miou=summary["best_validation_miou"])
                write_json(folder / "early_stopping_monitor.json", tracker.state_dict())
                if alive and tracker.stopped and summary["status"] == "running":
                    stop_training(read(folder / "launch.json"), folder, item["identity"])
                    summary, tracker = completed_tracker(folder, plan["early_stopping"])
                    summary.update(status="early_stopped", stopped_at=now(), stop_reason=tracker.stop_reason,
                                   early_stopping=tracker.state_dict(), stop_mechanism="external_completed_validation_json_monitor")
                    write_json(folder / "summary.json", summary)
                    alive = False
                if not alive and summary["status"] in TERMINAL:
                    item.update(phase="complete", completed_at=now())
            if not alive and item["phase"] != "complete":
                item.update(phase="failed", failed_at=now(), reason="Trainer exited without a terminal summary; inspect train.log. No automatic retry.")
        active = [state["runs"][name] for name in managed
                  if state["runs"][name]["phase"] == "running"]
        occupied = {v["gpu_uuid"] for v in active}
        available = [g for g in gpu_inventory() if str(g["index"]) in devices or g["uuid"] in devices]
        for entry in plan["runs"]:
            item = state["runs"][entry["name"]]
            if entry["name"] not in managed or item["phase"] != "queued" or len(active) >= max_parallel:
                continue
            candidates = [g for g in available if g["uuid"] not in occupied and g["free_mib"] >= minimum_free_mib]
            if not candidates:
                break
            gpu = candidates[0]
            for path, expected in plan["source_sha256"].items():
                if digest(resolve(path)) != expected:
                    raise RuntimeError(f"Source changed after freezing the run plan: {path}")
            config = resolve(entry["config"])
            if digest(config) != entry["config_sha256"]:
                raise RuntimeError(f"Configuration changed: {config}")
            training_settings(plan, entry["name"], read(config))
            folder = resolve(entry["output"])
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "last.pt").exists() or (folder / "launch.json").exists():
                raise FileExistsError(f"Refusing to overwrite an untracked existing run: {folder}")
            command = [sys.executable, "-B", "-u", "code/train.py", "--config", str(config)]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"], "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
            with (folder / "train.log").open("w") as log:
                process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            identity = process_identity(process.pid)
            if identity is None:
                raise RuntimeError("Trainer exited before its process identity could be captured")
            launch = {**identity, "command": command, "started_at": now(), "physical_gpu": gpu["index"],
                      "gpu_uuid": gpu["uuid"], "config_sha256": entry["config_sha256"],
                      "source_sha256": plan["source_sha256"], "plan": str(plan_path), "plan_sha256": state["plan_sha256"],
                      "execution_host": execution_host, "managed_runs": managed}
            write_json(folder / "launch.json", launch)
            children[entry["name"]] = process
            item.update(phase="running", identity=identity, gpu_uuid=gpu["uuid"], physical_gpu=gpu["index"],
                        started_at=launch["started_at"], execution_host=execution_host)
            active.append(item)
            occupied.add(gpu["uuid"])
            write_json(state_path, state)
            print(f"Started {entry['name']} on GPU {gpu['index']} (PID {process.pid})", flush=True)
        state["updated_at"] = now()
        write_json(state_path, state)
        comparison = analyze(plan, output)
        state["analysis_complete"] = comparison["complete"]
        if all(state["runs"][name]["phase"] in {"complete", "failed"} for name in managed):
            state["status"] = "complete" if all(state["runs"][name]["phase"] == "complete" for name in managed) else "failed"
            write_json(state_path, state)
            return 0 if state["status"] == "complete" else 1
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT / "result/prior_ablation")
    parser.add_argument("--devices", nargs="+", default=["0", "1", "2", "4"])
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--minimum-free-mib", type=int, default=14000)
    parser.add_argument("--managed-runs", nargs="+", default=None,
                        help="Run names owned by this host; other plan runs are observed through JSON only")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(read(args.plan), args.output)
        return 0
    return run(args.plan.resolve(), args.output.resolve(), args.devices, args.max_parallel,
               args.minimum_free_mib, managed_runs=args.managed_runs)


if __name__ == "__main__":
    raise SystemExit(main())
