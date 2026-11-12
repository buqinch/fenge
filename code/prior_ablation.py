"""Run the fixed prior ablations, monitor validation JSON, and compare results."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
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


def analyze(plan, output):
    """Use only complete validation records; never rank models on test results."""
    rows, identities, initial_hashes, histories, configurations = [], [], [], {}, {}
    for run in plan["runs"]:
        folder = resolve(run["output"])
        if not (folder / "summary.json").exists():
            continue
        summary = read(folder / "summary.json")
        config = read(folder / "run_config.json")
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
        directions = {}
        for truth, predicted in ((2, 0), (0, 2), (2, 3), (3, 1), (1, 3)):
            directions[f"{truth}->{predicted}"] = {
                "pixels": int(cm[truth, predicted]),
                "gt_conditional_rate": float(cm[truth, predicted] / cm[truth].sum()) if cm[truth].sum() else None}
        row = {"name": run["name"], "control": run.get("control"), "status": summary["status"],
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
    fields = ("epochs", "freeze_epochs", "batch_size", "accumulation", "head_lr", "backbone_lr", "prior_lr",
              "prior_seed", "prior_warmup_updates", "weight_decay", "gradient_clip", "seed", "horizontal_flip",
              "workers", "amp", "num_points", "val_batch_size", "val_tta", "val_tta_sizes")
    configs = list(configurations.values())
    for config in configs[1:]:
        for field in fields:
            if config.get(field) != configs[0].get(field):
                raise ValueError(f"Shared training/validation setting differs: {field}")
    if "geometry" in configurations and "geometry_control" in configurations:
        for field in ("geometry_paired_control", "gradient_checkpointing"):
            if configurations["geometry"].get(field) != configurations["geometry_control"].get(field):
                raise ValueError(f"Geometry and paired control differ: {field}")
    lookup = {row["name"]: row for row in rows}
    for row in rows:
        control = lookup.get(row["control"])
        row["delta_vs_control_pp"] = ((row["best_validation_miou"] - control["best_validation_miou"]) * 100
                                      if control else None)
        row["comparison_budget_equal"] = bool(control and row["global_update"] == control["global_update"])
        if control:
            common_epoch = min(row["completed_epochs"], control["completed_epochs"])
            first = max(histories[row["name"]][:common_epoch], key=lambda r: r["validation"]["miou"])
            second = max(histories[control["name"]][:common_epoch], key=lambda r: r["validation"]["miou"])
            row["matched_budget"] = {"through_epoch": common_epoch, "optimizer_updates": common_epoch * 1000,
                                     "best_miou": first["validation"]["miou"],
                                     "control_best_miou": second["validation"]["miou"],
                                     "delta_pp": 100 * (first["validation"]["miou"] - second["validation"]["miou"])}
            for a, b in zip(histories[row["name"]][:common_epoch], histories[control["name"]][:common_epoch]):
                for field in ("sample_order_sha256", "horizontal_flip_sha256", "sample_and_horizontal_flip_sha256"):
                    if not a.get(field) or a[field] != b.get(field):
                        raise ValueError(f"Matched training stream differs for {row['name']}: {field}")
                if row["name"] == "geometry" and a.get("geometry_view_sha256") != b.get("geometry_view_sha256"):
                    raise ValueError("Geometry and its control used different transforms")
    complete = len(rows) == len(plan["runs"]) and all(row["status"] in TERMINAL for row in rows)
    winner = max(rows, key=lambda r: r["best_validation_miou"]) if complete else None
    result = {"updated_at": now(), "complete": complete, "validation_only_selection": True,
              "initialization": "original_pretrained_backbones_and_identical_random_main_head",
              "common_initial_tensor_sha256": initial_hashes[0] if initial_hashes else None,
              "runs": rows, "selected_run": winner["name"] if winner else None,
              "interpretation": "Independent single-seed ablations. Shared maximum update budget and patience policy; actual early-stop epochs may differ. Paired geometry must be compared to geometry_control. No combined model has been tested."}
    write_json(output / "comparison.json", result)
    with (output / "comparison.csv").open("w", newline="") as stream:
        fields = ["name", "control", "status", "completed_epochs", "best_epoch", "global_update", "best_validation_miou", "delta_vs_control_pp"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 三种先验与高分辨率细节独立对照", "",
             "所有组从原始 DINOv3 ConvNeXt-Tiny＋ImageNet MobileNetV3-Large 开始，融合和区域头同种子重新初始化；没有加载第29轮分割权重。",
             "固定8000张训练（4000原图＋2000坑增强＋2000山岩增强），同一1601张验证，1024整图、microbatch 4×累积2，验证单视图。", "",
             "|实验|对照|状态|完成轮数|最佳轮次|验证mIoU|相对对照百分点|", "|---|---|---|---:|---:|---:|---:|"]
    for row in rows:
        delta = "—" if row["delta_vs_control_pp"] is None else f"{row['delta_vs_control_pp']:+.4f}"
        lines.append(f"|{row['name']}|{row['control'] or '—'}|{row['status']}|{row['completed_epochs']}|{row['best_epoch']}|{row['best_validation_miou']:.4%}|{delta}|")
    lines += ["", "亲和、中心—周围关系和细节注入分别与单视图control比较；几何一致性与同样两视图真值监督的geometry_control比较。",
              "最多30轮并使用相同早停规则；若实际更新数不同，不把最佳分差解释为固定训练预算下的纯模块因果收益。单种子结果尚不能证明稳定提升。",
              "完整类别IoU、精确率、召回率、六场景和混淆方向见[comparison.json](comparison.json)。"]
    if not complete:
        lines += ["", "**实验尚未全部结束，表中为阶段性结果，不据此选择最终方案。**"]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n")
    return result


def run(plan_path, output, devices, max_parallel, minimum_free_mib):
    plan = read(plan_path)
    if plan.get("status") != "ready" or not plan.get("source_sha256"):
        raise ValueError("Only a ready plan with nonempty frozen source hashes can launch training")
    if not devices or max_parallel < 1 or minimum_free_mib < 1:
        raise ValueError("Provide GPU devices and positive concurrency/memory limits")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / "state.json"
    state = read(state_path) if state_path.exists() else {"plan_sha256": digest(plan_path), "runs": {}, "status": "running"}
    if state["plan_sha256"] != digest(plan_path):
        raise ValueError("Runner state belongs to a different immutable plan")
    children = {}
    for entry in plan["runs"]:
        state["runs"].setdefault(entry["name"], {"phase": "queued"})
    while True:
        for entry in plan["runs"]:
            item = state["runs"][entry["name"]]
            folder = resolve(entry["output"])
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
        active = [v for v in state["runs"].values() if v["phase"] == "running"]
        occupied = {v["gpu_uuid"] for v in active}
        available = [g for g in gpu_inventory() if str(g["index"]) in devices or g["uuid"] in devices]
        for entry in plan["runs"]:
            item = state["runs"][entry["name"]]
            if item["phase"] != "queued" or len(active) >= max_parallel:
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
                      "source_sha256": plan["source_sha256"], "plan": str(plan_path), "plan_sha256": state["plan_sha256"]}
            write_json(folder / "launch.json", launch)
            children[entry["name"]] = process
            item.update(phase="running", identity=identity, gpu_uuid=gpu["uuid"], physical_gpu=gpu["index"], started_at=launch["started_at"])
            active.append(item)
            occupied.add(gpu["uuid"])
            write_json(state_path, state)
            print(f"Started {entry['name']} on GPU {gpu['index']} (PID {process.pid})", flush=True)
        state["updated_at"] = now()
        write_json(state_path, state)
        analyze(plan, output)
        if all(value["phase"] in {"complete", "failed"} for value in state["runs"].values()):
            state["status"] = "complete" if all(v["phase"] == "complete" for v in state["runs"].values()) else "failed"
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
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze(read(args.plan), args.output)
        return 0
    return run(args.plan.resolve(), args.output.resolve(), args.devices, args.max_parallel, args.minimum_free_mib)


if __name__ == "__main__":
    raise SystemExit(main())
