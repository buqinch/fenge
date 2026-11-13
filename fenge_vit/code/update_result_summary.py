#!/usr/bin/env python3
"""Summarize saved experiment results; no training, inference or publication.

Recompute scene scores from integer confusion matrices. Keep complete tests,
fixed difficult-subset diagnostics and the new selection validation separate.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
OUT = WORKSPACE / 'fenge_vit/result/progress_analysis'
SCENES = ('Moon_2', 'Moon_4', 'Moon_9')
FULL = dict(zip(SCENES, (1597, 1320, 2087)))
HARD = dict(zip(SCENES, (188, 187, 1125)))
CLASSES = ('regolith', 'rock', 'crater', 'mountain', 'sky')
RUNS = [
    ('CNN对照', 'fenge', 'dual_region_priors_control_8000_1024'),
    ('CNN细节注入', 'fenge', 'dual_region_priors_detail_8000_1024'),
    ('旧A融合', 'fenge_vit', 'a_fusion'),
    ('B Adapter', 'fenge_vit', 'b_adapter'),
    ('C单卡8000', 'fenge_vit', 'c_adapter_masking'),
    ('C四卡8000（已停）', 'fenge_vit', 'c_adapter_masking_103_4gpu'),
    ('C四卡6000', 'fenge_vit', 'c_adapter_masking_4000_1000_1000_103_4gpu'),
    ('旧纯ViT8000', 'fenge_vit_only', 'vit_only_m2f_8000_1024'),
    ('新A／困难验证', 'fenge_vit', 'a_fusion_pool8002_train6000_hardval1500_103_2gpu'),
    ('新纯ViT／困难验证', 'fenge_vit_only', 'vit_only_pool8002_train6000_hardval1500_103_2gpu'),
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(value):
    return '尚无' if value is None else f'{100 * value:.4f}%'


def link(path, label='指标'):
    return f'[{label}]({os.path.relpath(path, OUT)})'


def matrix_miou(matrix):
    if matrix.shape != (5, 5) or matrix.dtype.kind not in 'iu' or np.any(matrix < 0):
        raise ValueError('Expected nonnegative integer 5x5 confusion matrix')
    truth, prediction = matrix.sum(1), matrix.sum(0)
    union = truth + prediction - matrix.diagonal()
    values = np.divide(matrix.diagonal(), union, out=np.zeros(5), where=union > 0)
    return float(values[truth > 0].mean())


def measurement(path, name, scope):
    d = read(path)
    if d.get('partial') is not False or d.get('checkpoint_partial_training') is True:
        return None
    counts = FULL if scope == 'full' else HARD
    if d.get('image_count') != sum(counts.values()):
        raise ValueError(f'Incomplete coverage: {path}')
    scene_metrics, total = {}, np.zeros((5, 5), dtype=np.int64)
    for scene, count in counts.items():
        part = d['scenes'][scene]
        if part['image_count'] != count:
            raise ValueError(f'Scene image count differs: {path}/{scene}')
        matrix = np.asarray(part['confusion_matrix_gt_rows_pred_cols'])
        if matrix.sum() != count * 1024 * 1024:
            raise ValueError(f'Unexpected native pixel coverage: {path}/{scene}')
        score = matrix_miou(matrix)
        if not math.isclose(score, part['miou'], abs_tol=1e-12):
            raise ValueError(f'Scene score differs from confusion matrix: {path}')
        total += matrix
        scene_metrics[scene] = part
    if not np.array_equal(total, d['confusion_matrix_gt_rows_pred_cols']):
        raise ValueError(f'Pooled confusion matrix differs: {path}')
    pooled = matrix_miou(total)
    if not math.isclose(pooled, d['miou'], abs_tol=1e-12):
        raise ValueError(f'Pooled mIoU differs: {path}')
    score = sum(scene_metrics[s]['miou'] * w for s, w in zip(SCENES, (1, 1, 2))) / 4
    key = 'weighted_score' if scope == 'full' else 'hard_subset_score'
    if not math.isclose(score, d[key], abs_tol=1e-12):
        raise ValueError(f'Scene-weighted score differs: {path}')
    return {'experiment': name, 'scope': scope, 'checkpoint_epoch': d['checkpoint_epoch'],
            'checkpoint_sha256': d.get('checkpoint_sha256'), 'score': score,
            'pooled_miou': pooled, 'scenes': scene_metrics, 'path': str(path), 'sha256': sha(path),
            'tta_mode': d.get('tta_mode', d.get('tta')), 'tta_sizes': d.get('tta_sizes'),
            'tta_weights': d.get('tta_weights'), 'trigger_directory': path.parent.name}


def log_progress(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 20000))
        lines = stream.read().decode(errors='replace').splitlines()
    result = {'mtime': datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo('Asia/Shanghai')).isoformat(),
              'last_line': lines[-1] if lines else ''}
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict) and 'epoch' in row and 'batch' in row:
            result['training'] = {k: row[k] for k in ('epoch', 'batch', 'batches')}
            break
    result['validating'] = bool(lines and ('validation rank' in lines[-1] or lines[-1].startswith('Evaluation:')))
    return result


def collect():
    runs, measured = [], []
    for name, project, folder in RUNS:
        root = WORKSPACE / project / 'result' / folder
        summary_file = root / 'summary.json'
        summary = read(summary_file) if summary_file.exists() else {}
        config = read(root / 'run_config.json') if (root / 'run_config.json').exists() else {}
        hard_validation = 'hardval1500' in folder
        score = summary.get('best_validation_score') if hard_validation else summary.get('best_miou', summary.get('best_validation_miou'))
        run = {'experiment': name, 'project': project, 'run': folder,
               'validation_protocol': 'hard1500' if hard_validation else 'original1601',
               'status': summary.get('status', 'initializing'), 'completed_epochs': summary.get('completed_epochs', 0),
               'best_epoch': summary.get('best_epoch'), 'best_validation_score': score,
               'progress': log_progress(root / 'train.log'), 'summary_path': str(summary_file),
               'summary_sha256': sha(summary_file) if summary_file.exists() else None,
               'val_tta': config.get('val_tta'), 'train_images': summary.get('train_images'),
               'validation_images': summary.get('validation_images')}
        run['data_group'] = ('新6000／困难1500验证' if hard_validation else
                             '旧6000／原1601验证' if run['train_images'] == 6000 else
                             '旧8000／原1601验证')
        run['training_recipe'] = {key: config.get(key) for key in
            ('manifest', 'selection_manifest', 'variant', 'masking', 'batch_size',
             'accumulation', 'epochs', 'freeze_epochs', 'val_tta', 'val_tta_sizes', 'val_tta_weights')}
        if summary.get('best_epoch') is not None:
            val_path = root / f'validation_epoch_{summary["best_epoch"]:03d}.json'
            stored = read(val_path)
            val = stored.get('validation', stored)
            pooled = matrix_miou(np.asarray(val['confusion_matrix_gt_rows_pred_cols']))
            if not math.isclose(pooled, val['miou'], abs_tol=1e-12):
                raise ValueError(f'Validation pooled score differs: {val_path}')
            scene_scores = {}
            total = np.zeros((5, 5), dtype=np.int64)
            for scene, part in val['scenes'].items():
                matrix = np.asarray(part['confusion_matrix_gt_rows_pred_cols'])
                value = matrix_miou(matrix)
                if not math.isclose(value, part['miou'], abs_tol=1e-12):
                    raise ValueError(f'Validation scene score differs: {val_path}/{scene}')
                scene_scores[scene] = value
                total += matrix
            if not np.array_equal(total, val['confusion_matrix_gt_rows_pred_cols']):
                raise ValueError(f'Validation scene matrices disagree: {val_path}')
            chosen = sum(scene_scores[s] * w for s, w in zip(SCENES, (1, 1, 2))) / 4 if hard_validation else pooled
            if not math.isclose(chosen, score, abs_tol=1e-12):
                raise ValueError(f'Best summary differs from recomputed validation: {val_path}')
            run['best_validation_metrics'] = {'path':str(val_path), 'sha256':sha(val_path),
                'pooled_miou':pooled, 'scene_miou':scene_scores, 'score':chosen,
                'scenes':val['scenes'], 'class_iou':val['class_iou']}
        if hard_validation:
            history = []
            for path in sorted(root.glob('validation_epoch_*.json')):
                val = read(path)
                if val.get('epoch', 0) > run['completed_epochs']:
                    continue
                if val.get('partial') is not False or val.get('image_count') != 1500:
                    raise ValueError(f'Incomplete hard validation: {path}')
                scene_scores, total = {}, np.zeros((5, 5), dtype=np.int64)
                for scene, count in HARD.items():
                    part = val['scenes'][scene]
                    matrix = np.asarray(part['confusion_matrix_gt_rows_pred_cols'])
                    if part['image_count'] != count or matrix.sum() != count * 1024 * 1024:
                        raise ValueError(f'Invalid hard-validation coverage: {path}/{scene}')
                    value = matrix_miou(matrix)
                    if not math.isclose(value, part['miou'], abs_tol=1e-12):
                        raise ValueError(f'Hard-validation scene mIoU differs: {path}')
                    scene_scores[scene] = value
                    total += matrix
                weighted = sum(scene_scores[s] * w for s, w in zip(SCENES, (1, 1, 2))) / 4
                if (not np.array_equal(total, val['confusion_matrix_gt_rows_pred_cols'])
                        or not math.isclose(matrix_miou(total), val['miou'], abs_tol=1e-12)
                        or not math.isclose(weighted, val['validation_weighted_score'], abs_tol=1e-12)):
                    raise ValueError(f'Hard-validation aggregate disagrees: {path}')
                history.append({'epoch':val['epoch'], 'score':weighted, 'pooled_miou':val['miou'],
                    'scenes':val['scenes'], 'path':str(path), 'sha256':sha(path)})
            run['hard_validation_history'] = history
        all_paths = list(root.glob('periodic_test/*/test_metrics.json'))
        if project == 'fenge':
            all_paths += list(root.glob('test_epoch_*_multiscale_hflip_1024_768_metrics.json'))
        if folder == 'c_adapter_masking_4000_1000_1000_103_4gpu':
            all_paths += [WORKSPACE / 'fenge_vit/result/hard_test_1500_20261113/full_scan_metrics.json']
        full = [measurement(p, name, 'full') for p in sorted(all_paths)]
        hard = [measurement(p, name, 'hard') for p in sorted(root.glob('hard_periodic_test/*/hard_test_metrics.json'))]
        full, hard = [m for m in full if m], [m for m in hard if m]
        run['latest_full'] = max(full, key=lambda m: (m['checkpoint_epoch'], m['path'])) if full else None
        run['historical_max_full'] = max(full, key=lambda m: m['score']) if full else None
        run['latest_hard'] = max(hard, key=lambda m: (m['checkpoint_epoch'], m['path'])) if hard else None
        run['final_evaluation_jobs'] = [dict(read(p), source=str(p))
            for p in sorted(root.glob('periodic_test/final_epoch_*/status.json'))]
        run['all_hard'] = hard
        runs.append(run)
        measured.extend(full + hard)
    return runs, measured


def progress_text(run):
    progress = run['progress'] or {}
    batch = progress.get('training')
    if run['status'] == 'early_stopped':
        return f'第{run["completed_epochs"]}轮早停'
    if run['status'] == 'complete':
        return f'完成{run["completed_epochs"]}轮'
    if run['status'] == 'user_stopped':
        return f'已停止，完成{run["completed_epochs"]}轮'
    if progress.get('validating'):
        return (f'第{batch["epoch"]}轮验证中' if batch else '验证中')
    if batch:
        return f'第{batch["epoch"]}轮 {batch["batch"]}/{batch["batches"]}'
    return f'完成{run["completed_epochs"]}轮'


def active_tables(runs):
    active = [r for r in runs if r['status'] == 'training']
    lines = [f'仍在训练的实验共 **{len(active)} 组**。按已保存的训练状态分类；百分数后的轮次均指实际权重轮次。', '',
             'A＝ViT-S/16＋ConvNeXt-Tiny融合；B＝双骨干＋双向Adapter；C＝B＋随机输入遮挡训练；纯ViT＝仅ViT-S/16骨干。四类均使用Mask2Former解码器。', '']
    for group, description in (
        ('旧8000／原1601验证', '训练4000原图＋2000坑增强＋2000山岩增强，来自原6401训练来源；验证为原1601张。'),
        ('旧6000／原1601验证', '训练4000原图＋1000坑增强＋1000山岩增强，与旧8000组使用同一批4000原图；验证为原1601张。'),
        ('新6000／困难1500验证', '从Moon_1／3／5／6／7／8全部8002原图池重新抽4000张，另加1000坑增强＋1000山岩增强。验证为Moon_2／4／9固定困难188／187／1125张。'),
    ):
        members = [r for r in active if r['data_group'] == group]
        if not members:
            continue
        lines += [f'### {group}', '', description, '',
                  '|模型|当前进度|验证best（轮次）|最近困难1500（权重轮次）|最近完整5004 S（权重轮次）|',
                  '|---|---|---:|---:|---:|']
        for run in members:
            hard, full = run['latest_hard'], run['latest_full']
            hard_text = (f'{percent(hard["score"])}（{hard["checkpoint_epoch"]}）' if hard else
                         '同本行验证best' if run['validation_protocol'] == 'hard1500' else '尚无')
            full_text = f'{percent(full["score"])}（{full["checkpoint_epoch"]}）' if full else '尚未测试'
            lines.append(f'|{link(Path(run["summary_path"]),run["experiment"])}|{progress_text(run)}|'
                         f'{percent(run["best_validation_score"])}（{run["best_epoch"]}）|{hard_text}|{full_text}|')
        lines.append('')
    lines += ['旧1601验证为单1024、汇总混淆矩阵mIoU；困难1500和完整5004均为1024／768等权水平翻转、场景1∶1∶2加权。三个口径不能混比。完整测试成绩仅属于表中已测权重，不能套到当前新best。', '']
    for run in runs:
        pending = [j for j in run['final_evaluation_jobs'] if j.get('status') in ('queued', 'running')]
        if run['status'] != 'training' and pending:
            lines += [f'训练已结束、期末评测尚未完成：**{run["experiment"]}**（{run["data_group"]}），'
                      f'{progress_text(run)}；验证best为第{run["best_epoch"]}轮{percent(run["best_validation_score"])}。'
                      f'期末任务状态：{pending[-1]["status"]}，尚无新的完整测试分数。', '']
    return lines


def main():
    runs, measured = collect()
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    lines = ['# CNN与ViT最新结果', '', f'更新时间：**{timestamp}（北京时间）**。根据已落盘的完整指标统计，逐场景整数混淆矩阵已独立复算。', '',
             '## 正在训练：按模型与数据划分', ''] + active_tables(runs) + [
             '## 新A与纯ViT：6000训练／困难1500验证', '',
             '新验证集按Moon_2／4／9的188／187／1125张固定组成，以场景1∶1∶2加权选best。这里的验证分数与旧1601张验证mIoU、完整5004张测试S分别报告。', '',
             '|实验|进度|验证best轮次|困难验证加权分数|', '|---|---|---:|---:|']
    for run in runs:
        if run['validation_protocol'] == 'hard1500':
            lines.append(f'|{run["experiment"]}|{progress_text(run)}|{run["best_epoch"] or "尚无"}|{percent(run["best_validation_score"])}|')
    baseline_path = WORKSPACE / 'fenge_vit/result/hard_test_1500_20261113/source_hard_subset_metrics.json'
    baseline = measurement(baseline_path, '固定困难集来源基准C6000', 'hard')
    lines += ['', f'固定来源baseline为C四卡6000第18轮：**{percent(baseline["score"])}**，详见{link(baseline_path)}。它是定义这批困难图时冻结的模型，同样采用1024／768等权翻转TTA；当前已测模型最高值另外统计。', '',
              '|新实验|验证轮次|Moon_2|Moon_4|Moon_9|加权分数|较上次验证变化（百分点）|',
              '|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        previous = None
        for row in run.get('hard_validation_history', []):
            values = '|'.join(percent(row['scenes'][s]['miou']) for s in SCENES)
            delta = '—' if previous is None else f'{100*(row["score"]-previous):+.4f}'
            lines.append(f'|{link(Path(row["path"]),run["experiment"])}|{row["epoch"]}|{values}|{percent(row["score"])}|{delta}|')
            previous = row['score']
    lines += ['', '|新实验best|场景|月壤IoU|岩石IoU|撞击坑IoU|山体IoU|天空IoU|',
              '|---|---|---:|---:|---:|---:|---:|']
    for run in runs:
        if run['validation_protocol'] == 'hard1500' and run.get('best_validation_metrics'):
            for scene, metrics in run['best_validation_metrics']['scenes'].items():
                values = '|'.join('不计分' if metrics['class_iou'][c] is None else percent(metrics['class_iou'][c]) for c in CLASSES)
                lines.append(f'|{run["experiment"]}／{run["best_epoch"]}轮|{scene}|{values}|')
    lines += ['', '每2轮及计划末轮验证；最多30轮，5次有效验证未显著提升则早停。新A在103 GPU4／5，纯ViT在6／7，均每卡batch4、有效batch8。尚未完成验证的轮次不提前报分。', '',
              '## 原验证协议的训练进展', '', '|实验|进度|最佳轮次|1601张验证best mIoU|', '|---|---|---:|---:|']
    for run in runs:
        if run['validation_protocol'] == 'original1601':
            lines.append(f'|{run["experiment"]}|{progress_text(run)}|{run["best_epoch"] or "—"}|{percent(run["best_validation_score"])}|')
    lines += ['', '## 最新已完成的完整Moon_2／4／9测试', '',
              '每份覆盖1597／1320／2087张，共5004张，使用1024／768等权水平翻转4视图。S=(Moon_2+Moon_4+2×Moon_9)/4；列出的权重轮次来自checkpoint_epoch，不能把已有测试套到尚未评测的新best。', '',
              '|实验|已测权重轮次|Moon_2|Moon_4|Moon_9|最新S|历史已测最高S（轮次）|', '|---|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        m, historical = run['latest_full'], run['historical_max_full']
        if m:
            values = '|'.join(percent(m['scenes'][s]['miou']) for s in SCENES)
            lines.append(f'|{link(Path(m["path"]),run["experiment"])}|{m["checkpoint_epoch"]}|{values}|{percent(m["score"])}|{percent(historical["score"])}（{historical["checkpoint_epoch"]}）|')
    lines += ['', 'C四卡6000的第18轮结果来自构建困难集时对完整5004张的冻结权重扫描。历史最高只表示已经观察到的测试值，不代表按测试回选的最终权重。当前完整测试记录中' + ('已有' if any(m['scope']=='full' and m['score'] > .92 for m in measured) else '没有') + '加权S超过92%的结果。', '',
              '## 原模型在固定困难1500张上的诊断', '',
              '这里仍由原1601张验证集选择权重，困难1500只用于诊断；新A／纯ViT则用相同1500参与选轮，两种用途不同。', '',
              '|实验|已测权重轮次|Moon_2|Moon_4|Moon_9|困难集加权分数|', '|---|---:|---:|---:|---:|---:|']
    for run in runs:
        m = run['latest_hard']
        if m:
            values = '|'.join(percent(m['scenes'][s]['miou']) for s in SCENES)
            lines.append(f'|{link(Path(m["path"]),run["experiment"])}|{m["checkpoint_epoch"]}|{values}|{percent(m["score"])}|')
    lines += ['', '困难集Moon_4没有山体GT，因此按其余四类计算场景mIoU，山体IoU为null。1500的分数不能与5004张S直接相减。', '',
              '## 当前判断', '',
              '- 旧A第14轮曾达到90.6628%，最终验证best第21轮为89.7335%，验证提高没有保证测试同步提高。',
              '- C四卡6000第15轮为90.6230%，第18轮完整扫描为90.0233%；仅凭旧1601张验证mIoU无法判断2／4／9上的收益。',
              '- 新A与纯ViT的验证趋势和逐类成绩列于上表；前3轮冻结骨干，第4轮才开始联合微调，阶段差距不等同于最终上限。',
              '- 新协议把原测试1500张用于验证。期末会同时报告完整5004张和剩余3504张，前者包含验证，后者也参与过难度筛选，均不称为未经查看的独立留出集。', '',
              '[机器可读快照及来源SHA](latest_status.json) · [已完成评测逐场景五类指标](completed_test_classes.csv)。']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'latest_report_zh.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'latest_status.json').write_text(json.dumps({'schema':'lusnar_result_snapshot_v2', 'updated_at':timestamp,
        'metrics_verified_from_integer_confusion_matrices':True, 'runs':runs, 'fixed_hard_baseline':baseline,
        'active_training_runs':[r['run'] for r in runs if r['status'] == 'training'],
        'completed_measurements':measured}, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'completed_test_classes.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['experiment','scope','checkpoint_epoch','scene','class','iou','precision','recall','score','source'])
        for m in measured:
            for scene, part in m['scenes'].items():
                for c in CLASSES:
                    writer.writerow([m['experiment'],m['scope'],m['checkpoint_epoch'],scene,c,
                        part['class_iou'][c],part['class_precision'][c],part['class_recall'][c],m['score'],m['path']])
    print(json.dumps({'updated_at':timestamp,'runs':len(runs),'verified_completed_metrics':len(measured),
                      'report':str(OUT/'latest_report_zh.md')},ensure_ascii=False))


if __name__ == '__main__':
    main()
