#!/usr/bin/env python3
"""
Ablation Test HTML Report Generator.

Collects ablation experiment results from S3 and produces a standalone
HTML report with dark-theme CSS visualisation (no JavaScript required).

Usage::

    python scripts/generate_ablation_report.py \
        --s3-base s3://aiops-ple-financial/ablation-test/20260320-143000 \
        --output docs/ablation_report_20260320.html

The report includes:
    1. Experiment Overview
    2. Data Quality
    3. Feature Group Ablation Results
    4. Expert Ablation Results
    5. Hyperparameter Sensitivity
    6. Best Config Full Pipeline
    7. Per-Task Details
    8. Reason Generation Validation
    9. Pipeline Details
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("generate_ablation_report")

REGION = "ap-northeast-2"
S3_BUCKET = "aiops-ple-financial"

# Task definitions from the pipeline config
TASK_NAMES = [
    "is_fraud", "will_transact", "churn", "retention", "life_stage", "ltv",
    "balance_util", "engagement", "channel", "timing", "nba",
    "spending_category", "consumption_cycle", "spending_bucket",
    "merchant_affinity", "brand_prediction",
]

TASK_TYPES = {
    "is_fraud": "binary", "will_transact": "binary", "churn": "binary",
    "retention": "binary", "life_stage": "multiclass", "ltv": "regression",
    "balance_util": "regression", "engagement": "regression",
    "channel": "multiclass", "timing": "multiclass", "nba": "multiclass",
    "spending_category": "multiclass", "consumption_cycle": "multiclass",
    "spending_bucket": "regression", "merchant_affinity": "regression",
    "brand_prediction": "multiclass",
}

PRIMARY_METRICS = {
    "binary": "auc_roc",
    "multiclass": "f1_macro",
    "regression": "rmse",
}


# ===================================================================
# S3 data collection
# ===================================================================

def _s3_client():
    import boto3
    return boto3.client("s3", region_name=REGION)


def _download_json(s3_uri: str) -> Optional[dict]:
    """Download and parse a JSON file from S3."""
    try:
        s3 = _s3_client()
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.debug("Failed to download %s: %s", s3_uri, e)
        return None


def _list_keys(prefix: str) -> List[str]:
    """List S3 keys under a prefix."""
    s3 = _s3_client()
    keys = []
    s3_prefix = prefix.replace(f"s3://{S3_BUCKET}/", "")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def collect_phase_metrics(
    s3_base: str,
    phase: str,
    scenarios: List[str],
) -> Dict[str, Optional[dict]]:
    """Collect eval_metrics.json for each scenario in a phase.

    Returns
    -------
    dict
        Mapping scenario_name -> metrics dict (or None if missing).
    """
    results: Dict[str, Optional[dict]] = {}
    for scenario in scenarios:
        # Try multiple possible output paths
        for suffix in [
            "output/eval_metrics.json",
            "eval_metrics.json",
            "output/model/eval_metrics.json",
        ]:
            uri = f"{s3_base}/{phase}/{scenario}/{suffix}"
            data = _download_json(uri)
            if data is not None:
                results[scenario] = data
                break
        else:
            results[scenario] = None
    return results


def collect_training_metrics(
    s3_base: str,
    phase: str,
    scenario: str,
) -> Optional[dict]:
    """Collect training_metrics.json for a specific scenario."""
    for suffix in [
        "output/training_metrics.json",
        "training_metrics.json",
        "output/model/training_metrics.json",
    ]:
        uri = f"{s3_base}/{phase}/{scenario}/{suffix}"
        data = _download_json(uri)
        if data is not None:
            return data
    return None


def collect_manifest(s3_base: str) -> Optional[dict]:
    """Load the orchestration manifest."""
    return _download_json(f"{s3_base}/manifest.json")


# ===================================================================
# CSS / HTML building blocks
# ===================================================================

DARK_CSS = """
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --accent-blue: #58a6ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-orange: #d29922;
    --accent-purple: #bc8cff;
    --accent-cyan: #39d2c0;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
    padding: 20px;
    max-width: 1400px;
    margin: 0 auto;
}
h1 { font-size: 1.8em; margin: 20px 0 10px; color: var(--accent-blue); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
h2 { font-size: 1.4em; margin: 24px 0 12px; color: var(--accent-cyan); }
h3 { font-size: 1.1em; margin: 16px 0 8px; color: var(--text-primary); }
p { margin: 8px 0; color: var(--text-secondary); }
a { color: var(--accent-blue); text-decoration: none; }
a:hover { text-decoration: underline; }

.card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
    margin: 12px 0;
}
.card-header {
    font-weight: 600;
    font-size: 1.05em;
    color: var(--text-primary);
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Overview stats grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin: 12px 0;
}
.stat-box {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
}
.stat-value {
    font-size: 1.6em;
    font-weight: 700;
    color: var(--accent-blue);
}
.stat-label {
    font-size: 0.85em;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 8px 0;
    font-size: 0.9em;
}
th, td {
    padding: 8px 10px;
    border: 1px solid var(--border);
    text-align: left;
}
th {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    font-weight: 600;
    position: sticky;
    top: 0;
}
td { color: var(--text-secondary); }
tr:hover td { background: rgba(88,166,255,0.05); }
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Heatmap cells */
.heat-best  { background: rgba(63,185,80,0.25); color: var(--accent-green); font-weight: 600; }
.heat-good  { background: rgba(63,185,80,0.10); color: var(--accent-green); }
.heat-mid   { background: rgba(210,153,34,0.10); color: var(--accent-orange); }
.heat-bad   { background: rgba(248,81,73,0.10); color: var(--accent-red); }
.heat-worst { background: rgba(248,81,73,0.25); color: var(--accent-red); font-weight: 600; }

/* CSS bar chart */
.bar-container { width: 100%; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden; height: 22px; margin: 4px 0; }
.bar {
    height: 100%;
    border-radius: 3px;
    display: flex;
    align-items: center;
    padding-left: 6px;
    font-size: 0.8em;
    color: var(--text-primary);
    white-space: nowrap;
    transition: width 0.3s ease;
}
.bar-blue   { background: var(--accent-blue); }
.bar-green  { background: var(--accent-green); }
.bar-red    { background: var(--accent-red); }
.bar-orange { background: var(--accent-orange); }
.bar-purple { background: var(--accent-purple); }
.bar-cyan   { background: var(--accent-cyan); }

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 600;
}
.badge-pass { background: rgba(63,185,80,0.15); color: var(--accent-green); }
.badge-fail { background: rgba(248,81,73,0.15); color: var(--accent-red); }
.badge-warn { background: rgba(210,153,34,0.15); color: var(--accent-orange); }
.badge-info { background: rgba(88,166,255,0.15); color: var(--accent-blue); }

/* Collapsible sections */
details { margin: 8px 0; }
summary {
    cursor: pointer;
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-weight: 600;
}
summary:hover { background: rgba(88,166,255,0.05); }
details[open] summary { border-radius: 6px 6px 0 0; }
details > div {
    border: 1px solid var(--border);
    border-top: none;
    border-radius: 0 0 6px 6px;
    padding: 12px;
    background: var(--bg-secondary);
}

/* Sensitivity chart (pure CSS) */
.chart-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 4px 0;
}
.chart-label { width: 80px; text-align: right; font-size: 0.85em; color: var(--text-secondary); flex-shrink: 0; }
.chart-bar-wrap { flex: 1; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden; height: 18px; }
.chart-bar {
    height: 100%;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 6px;
    font-size: 0.75em;
    color: var(--text-primary);
}

/* Timeline */
.timeline { position: relative; padding-left: 24px; margin: 12px 0; }
.timeline::before {
    content: '';
    position: absolute;
    left: 8px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
}
.timeline-item {
    position: relative;
    margin: 12px 0;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
}
.timeline-item::before {
    content: '';
    position: absolute;
    left: -20px;
    top: 14px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-blue);
    border: 2px solid var(--bg-primary);
}
.timeline-item.completed::before { background: var(--accent-green); }
.timeline-item.failed::before { background: var(--accent-red); }

/* Footer */
.footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85em;
}
"""


def _html_escape(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _format_number(val: Any, decimals: int = 4) -> str:
    """Format a number for display."""
    if val is None:
        return "-"
    try:
        v = float(val)
        if abs(v) >= 1000:
            return f"{v:,.{min(2, decimals)}f}"
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _heat_class(val: float, values: List[float], higher_is_better: bool = True) -> str:
    """Assign a heatmap CSS class based on rank within values."""
    if not values or val is None:
        return ""
    sorted_vals = sorted(v for v in values if v is not None)
    if not sorted_vals:
        return ""
    n = len(sorted_vals)
    try:
        rank = sorted_vals.index(val)
    except ValueError:
        return ""
    if not higher_is_better:
        rank = n - 1 - rank
    pct = rank / max(n - 1, 1)
    if pct >= 0.9:
        return "heat-best"
    if pct >= 0.6:
        return "heat-good"
    if pct >= 0.4:
        return "heat-mid"
    if pct >= 0.1:
        return "heat-bad"
    return "heat-worst"


def _bar_html(value: float, max_value: float, label: str = "", css_class: str = "bar-blue") -> str:
    """Render a CSS bar chart row."""
    pct = min(100, max(0, (value / max_value) * 100)) if max_value > 0 else 0
    lbl = label or _format_number(value)
    return (
        f'<div class="bar-container">'
        f'<div class="bar {css_class}" style="width:{pct:.1f}%">{_html_escape(lbl)}</div>'
        f'</div>'
    )


# ===================================================================
# Section generators
# ===================================================================

def section_overview(manifest: Optional[dict], s3_base: str) -> str:
    """Section 1: Experiment Overview."""
    ts = s3_base.rstrip("/").split("/")[-1] if s3_base else "unknown"
    phases = manifest.get("phases", {}) if manifest else {}

    total_jobs = 0
    for k, v in phases.items():
        if isinstance(v, list):
            total_jobs += len(v)
        else:
            total_jobs += 1

    return f"""
    <h1>1. Experiment Overview</h1>
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-value">ealtman2019</div>
            <div class="stat-label">Dataset</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">16</div>
            <div class="stat-label">Tasks</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">~469</div>
            <div class="stat-label">Features</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">2,000</div>
            <div class="stat-label">Users</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">28</div>
            <div class="stat-label">Ablation Scenarios</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{total_jobs}</div>
            <div class="stat-label">SageMaker Jobs</div>
        </div>
    </div>
    <div class="card">
        <div class="card-header">Experiment Details</div>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Timestamp</td><td>{_html_escape(ts)}</td></tr>
            <tr><td>S3 Base</td><td><code>{_html_escape(s3_base)}</code></td></tr>
            <tr><td>Feature Group Scenarios</td><td>9 (full + 8 ablations)</td></tr>
            <tr><td>Expert Ablation Scenarios</td><td>7 (full_basket + 5 removals + mlp_only)</td></tr>
            <tr><td>Hyperparameter Scenarios</td><td>12 (4 LR + 4 Temp + 3 Layers + 1 ExtDim)</td></tr>
            <tr><td>Region</td><td>ap-northeast-2</td></tr>
        </table>
    </div>
    """


def section_data_quality(s3_base: str) -> str:
    """Section 2: Data Quality."""
    # Try to load quality gate results
    qg_data = _download_json(f"{s3_base}/phase0/data/output/quality_gate_report.json")
    feature_stats = _download_json(f"{s3_base}/phase0/data/output/feature_stats.json")

    html = '<h1>2. Data Quality</h1>\n'

    # Quality Gate
    html += '<div class="card"><div class="card-header">Quality Gate Results</div>\n'
    if qg_data:
        verdict = qg_data.get("verdict", "unknown")
        badge_cls = "badge-pass" if verdict == "pass" else "badge-fail"
        html += f'<p>Verdict: <span class="badge {badge_cls}">{_html_escape(verdict).upper()}</span></p>\n'
        checks = qg_data.get("checks", [])
        if checks:
            html += '<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>\n'
            for c in checks:
                st = c.get("status", "unknown")
                badge = "badge-pass" if st == "pass" else "badge-fail"
                html += (
                    f'<tr><td>{_html_escape(c.get("name", ""))}</td>'
                    f'<td><span class="badge {badge}">{st}</span></td>'
                    f'<td>{_html_escape(c.get("detail", ""))}</td></tr>\n'
                )
            html += '</table>\n'
    else:
        html += '<p>Quality gate report not available.</p>\n'
    html += '</div>\n'

    # Feature stats
    html += '<div class="card"><div class="card-header">Feature Statistics</div>\n'
    if feature_stats:
        stats = feature_stats.get("features", {})
        html += '<table><tr><th>Feature Group</th><th>Count</th><th>Mean</th><th>Std</th><th>Null %</th></tr>\n'
        for group_name, group_stats in stats.items():
            count = group_stats.get("count", 0)
            mean = _format_number(group_stats.get("mean"), 3)
            std = _format_number(group_stats.get("std"), 3)
            null_pct = _format_number(group_stats.get("null_pct", 0), 1)
            html += (
                f'<tr><td>{_html_escape(group_name)}</td>'
                f'<td class="num">{count}</td>'
                f'<td class="num">{mean}</td>'
                f'<td class="num">{std}</td>'
                f'<td class="num">{null_pct}%</td></tr>\n'
            )
        html += '</table>\n'
    else:
        html += '<p>Feature statistics not available (run Phase 0 first).</p>\n'
    html += '</div>\n'

    # Label distribution
    label_dist = _download_json(f"{s3_base}/phase0/data/output/label_distribution.json")
    html += '<div class="card"><div class="card-header">Label Distribution</div>\n'
    if label_dist:
        for task_name in TASK_NAMES:
            dist = label_dist.get(task_name, {})
            if not dist:
                continue
            html += f'<h3>{_html_escape(task_name)} ({TASK_TYPES.get(task_name, "?")})</h3>\n'
            total = sum(float(v) for v in dist.values()) or 1
            max_val = max(float(v) for v in dist.values()) or 1
            for label_val, count in dist.items():
                pct = float(count) / total * 100
                html += (
                    f'<div class="chart-row">'
                    f'<span class="chart-label">{_html_escape(str(label_val))}</span>'
                    f'<div class="chart-bar-wrap">'
                    f'<div class="chart-bar bar-blue" style="width:{float(count)/max_val*100:.1f}%">'
                    f'{count} ({pct:.1f}%)</div></div></div>\n'
                )
    else:
        html += '<p>Label distribution data not available.</p>\n'
    html += '</div>\n'

    return html


def section_feature_ablation(s3_base: str) -> str:
    """Section 3: Feature Group Ablation Results."""
    scenarios = [
        "full", "no_tda", "no_temporal", "no_graph", "no_economics",
        "no_multidisciplinary", "no_hmm", "no_merchant", "base_only",
    ]
    metrics_map = collect_phase_metrics(s3_base, "phase1", scenarios)

    html = '<h1>3. Feature Group Ablation Results</h1>\n'

    # Heatmap matrix: scenarios x tasks
    html += '<div class="card"><div class="card-header">Scenario x Task Heatmap (Primary Metric)</div>\n'
    html += '<div style="overflow-x: auto;">\n'
    html += '<table><tr><th>Scenario</th>'
    for t in TASK_NAMES:
        html += f'<th>{_html_escape(t)}</th>'
    html += '<th>Avg</th></tr>\n'

    # Collect all values per task for heatmap coloring
    task_values: Dict[str, List[float]] = {t: [] for t in TASK_NAMES}
    scenario_data: Dict[str, Dict[str, Optional[float]]] = {}

    for sc_name in scenarios:
        m = metrics_map.get(sc_name)
        row: Dict[str, Optional[float]] = {}
        for t in TASK_NAMES:
            task_type = TASK_TYPES[t]
            primary = PRIMARY_METRICS[task_type]
            val = None
            if m:
                task_metrics = m.get("per_task", {}).get(t, {})
                val = task_metrics.get(primary)
            row[t] = val
            if val is not None:
                task_values[t].append(val)
        scenario_data[sc_name] = row

    for sc_name in scenarios:
        row = scenario_data[sc_name]
        html += f'<tr><td><strong>{_html_escape(sc_name)}</strong></td>'
        vals = []
        for t in TASK_NAMES:
            val = row[t]
            higher_is_better = TASK_TYPES[t] != "regression"
            cls = ""
            if val is not None:
                cls = _heat_class(val, task_values[t], higher_is_better)
                vals.append(val)
            html += f'<td class="num {cls}">{_format_number(val)}</td>'
        avg = sum(vals) / len(vals) if vals else None
        html += f'<td class="num"><strong>{_format_number(avg)}</strong></td></tr>\n'

    html += '</table></div></div>\n'

    # Feature group contribution (full - no_X)
    html += '<div class="card"><div class="card-header">Feature Group Contribution (delta from full)</div>\n'
    full_row = scenario_data.get("full", {})
    contributions: List[Tuple[str, float]] = []

    html += '<table><tr><th>Feature Group</th><th>Avg Delta</th><th>Impact</th></tr>\n'
    for sc_name in scenarios:
        if sc_name == "full" or sc_name == "base_only":
            continue
        row = scenario_data[sc_name]
        deltas = []
        for t in TASK_NAMES:
            full_val = full_row.get(t)
            ablated_val = row.get(t)
            if full_val is not None and ablated_val is not None:
                higher_is_better = TASK_TYPES[t] != "regression"
                if higher_is_better:
                    delta = full_val - ablated_val  # positive = feature was helpful
                else:
                    delta = ablated_val - full_val  # positive (higher RMSE) = feature was helpful
                deltas.append(delta)
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        group_name = sc_name.replace("no_", "")
        contributions.append((group_name, avg_delta))

        # Bar
        color = "bar-green" if avg_delta > 0 else "bar-red"
        bar_val = abs(avg_delta)
        html += (
            f'<tr><td>{_html_escape(group_name)}</td>'
            f'<td class="num">{avg_delta:+.4f}</td>'
            f'<td>{_bar_html(bar_val, 0.1, f"{avg_delta:+.4f}", color)}</td></tr>\n'
        )

    html += '</table></div>\n'

    # Top 3 per task
    html += '<div class="card"><div class="card-header">Top 3 Most Important Feature Groups Per Task</div>\n'
    html += '<table><tr><th>Task</th><th>#1</th><th>#2</th><th>#3</th></tr>\n'
    for t in TASK_NAMES:
        higher_is_better = TASK_TYPES[t] != "regression"
        task_contribs = []
        for sc_name in scenarios:
            if sc_name in ("full", "base_only"):
                continue
            full_val = full_row.get(t)
            ablated_val = scenario_data[sc_name].get(t)
            if full_val is not None and ablated_val is not None:
                if higher_is_better:
                    delta = full_val - ablated_val
                else:
                    delta = ablated_val - full_val
                group_name = sc_name.replace("no_", "")
                task_contribs.append((group_name, delta))
        task_contribs.sort(key=lambda x: x[1], reverse=True)
        top3 = task_contribs[:3]
        cells = ""
        for i in range(3):
            if i < len(top3):
                cells += f'<td>{_html_escape(top3[i][0])} ({top3[i][1]:+.4f})</td>'
            else:
                cells += '<td>-</td>'
        html += f'<tr><td>{_html_escape(t)}</td>{cells}</tr>\n'
    html += '</table></div>\n'

    return html


def section_expert_ablation(s3_base: str) -> str:
    """Section 4: Expert Ablation Results."""
    scenarios = [
        "full_basket", "no_deepfm", "no_temporal", "no_hgcn",
        "no_perslay", "no_causal", "mlp_only",
    ]
    metrics_map = collect_phase_metrics(s3_base, "phase2", scenarios)

    html = '<h1>4. Expert Ablation Results</h1>\n'

    # Heatmap matrix
    html += '<div class="card"><div class="card-header">Scenario x Task Heatmap (Primary Metric)</div>\n'
    html += '<div style="overflow-x: auto;">\n'
    html += '<table><tr><th>Scenario</th>'
    for t in TASK_NAMES:
        html += f'<th>{_html_escape(t)}</th>'
    html += '<th>Avg</th></tr>\n'

    task_values: Dict[str, List[float]] = {t: [] for t in TASK_NAMES}
    scenario_data: Dict[str, Dict[str, Optional[float]]] = {}

    for sc_name in scenarios:
        m = metrics_map.get(sc_name)
        row: Dict[str, Optional[float]] = {}
        for t in TASK_NAMES:
            task_type = TASK_TYPES[t]
            primary = PRIMARY_METRICS[task_type]
            val = None
            if m:
                task_metrics = m.get("per_task", {}).get(t, {})
                val = task_metrics.get(primary)
            row[t] = val
            if val is not None:
                task_values[t].append(val)
        scenario_data[sc_name] = row

    for sc_name in scenarios:
        row = scenario_data[sc_name]
        html += f'<tr><td><strong>{_html_escape(sc_name)}</strong></td>'
        vals = []
        for t in TASK_NAMES:
            val = row[t]
            higher_is_better = TASK_TYPES[t] != "regression"
            cls = ""
            if val is not None:
                cls = _heat_class(val, task_values[t], higher_is_better)
                vals.append(val)
            html += f'<td class="num {cls}">{_format_number(val)}</td>'
        avg = sum(vals) / len(vals) if vals else None
        html += f'<td class="num"><strong>{_format_number(avg)}</strong></td></tr>\n'

    html += '</table></div></div>\n'

    # Expert contribution
    html += '<div class="card"><div class="card-header">Expert Contribution (delta from full_basket)</div>\n'
    full_row = scenario_data.get("full_basket", {})

    html += '<table><tr><th>Expert</th><th>Avg Delta</th><th>Impact</th></tr>\n'
    for sc_name in scenarios:
        if sc_name in ("full_basket", "mlp_only"):
            continue
        row = scenario_data[sc_name]
        deltas = []
        for t in TASK_NAMES:
            full_val = full_row.get(t)
            ablated_val = row.get(t)
            if full_val is not None and ablated_val is not None:
                higher_is_better = TASK_TYPES[t] != "regression"
                if higher_is_better:
                    delta = full_val - ablated_val
                else:
                    delta = ablated_val - full_val
                deltas.append(delta)
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        expert_name = sc_name.replace("no_", "")
        color = "bar-green" if avg_delta > 0 else "bar-red"
        bar_val = abs(avg_delta)
        html += (
            f'<tr><td>{_html_escape(expert_name)}</td>'
            f'<td class="num">{avg_delta:+.4f}</td>'
            f'<td>{_bar_html(bar_val, 0.1, f"{avg_delta:+.4f}", color)}</td></tr>\n'
        )
    html += '</table></div>\n'

    # Top 3 experts per task
    html += '<div class="card"><div class="card-header">Top 3 Most Important Experts Per Task</div>\n'
    html += '<table><tr><th>Task</th><th>#1</th><th>#2</th><th>#3</th></tr>\n'
    for t in TASK_NAMES:
        higher_is_better = TASK_TYPES[t] != "regression"
        task_contribs = []
        for sc_name in scenarios:
            if sc_name in ("full_basket", "mlp_only"):
                continue
            full_val = full_row.get(t)
            ablated_val = scenario_data[sc_name].get(t)
            if full_val is not None and ablated_val is not None:
                if higher_is_better:
                    delta = full_val - ablated_val
                else:
                    delta = ablated_val - full_val
                expert_name = sc_name.replace("no_", "")
                task_contribs.append((expert_name, delta))
        task_contribs.sort(key=lambda x: x[1], reverse=True)
        top3 = task_contribs[:3]
        cells = ""
        for i in range(3):
            if i < len(top3):
                cells += f'<td>{_html_escape(top3[i][0])} ({top3[i][1]:+.4f})</td>'
            else:
                cells += '<td>-</td>'
        html += f'<tr><td>{_html_escape(t)}</td>{cells}</tr>\n'
    html += '</table></div>\n'

    return html


def section_hp_sensitivity(s3_base: str) -> str:
    """Section 5: Hyperparameter Sensitivity."""
    html = '<h1>5. Hyperparameter Sensitivity</h1>\n'

    # Learning rate sweep
    lr_scenarios = [f"lr_{lr}" for lr in [0.0001, 0.0005, 0.001, 0.005]]
    lr_metrics = collect_phase_metrics(s3_base, "phase3", lr_scenarios)

    html += '<div class="card"><div class="card-header">Learning Rate Sensitivity</div>\n'
    html += '<table><tr><th>Learning Rate</th><th>Avg Primary Metric</th><th>Performance</th></tr>\n'
    lr_scores = []
    for sc_name in lr_scenarios:
        m = lr_metrics.get(sc_name)
        score = m.get("aggregate_score", 0) if m else 0
        lr_scores.append(score)
    max_lr_score = max(lr_scores) if lr_scores and max(lr_scores) > 0 else 1

    for sc_name, score in zip(lr_scenarios, lr_scores):
        lr_val = sc_name.replace("lr_", "")
        html += (
            f'<tr><td>{lr_val}</td>'
            f'<td class="num">{_format_number(score)}</td>'
            f'<td>{_bar_html(score, max_lr_score, _format_number(score), "bar-blue")}</td></tr>\n'
        )
    html += '</table></div>\n'

    # Temperature sweep
    temp_scenarios = [f"temp_{t}" for t in [1.0, 3.0, 5.0, 10.0]]
    temp_metrics = collect_phase_metrics(s3_base, "phase3", temp_scenarios)

    html += '<div class="card"><div class="card-header">Distillation Temperature Sensitivity</div>\n'
    html += '<table><tr><th>Temperature</th><th>Avg Fidelity</th><th>Performance</th></tr>\n'
    temp_scores = []
    for sc_name in temp_scenarios:
        m = temp_metrics.get(sc_name)
        score = m.get("aggregate_score", 0) if m else 0
        temp_scores.append(score)
    max_temp_score = max(temp_scores) if temp_scores and max(temp_scores) > 0 else 1

    for sc_name, score in zip(temp_scenarios, temp_scores):
        temp_val = sc_name.replace("temp_", "")
        html += (
            f'<tr><td>{temp_val}</td>'
            f'<td class="num">{_format_number(score)}</td>'
            f'<td>{_bar_html(score, max_temp_score, _format_number(score), "bar-cyan")}</td></tr>\n'
        )
    html += '</table></div>\n'

    # Num layers sweep
    layer_scenarios = [f"layers_{n}" for n in [1, 2, 3]]
    layer_metrics = collect_phase_metrics(s3_base, "phase3", layer_scenarios)

    html += '<div class="card"><div class="card-header">PLE Layers Sensitivity</div>\n'
    html += '<table><tr><th>Num Layers</th><th>Avg Primary Metric</th><th>Performance</th></tr>\n'
    layer_scores = []
    for sc_name in layer_scenarios:
        m = layer_metrics.get(sc_name)
        score = m.get("aggregate_score", 0) if m else 0
        layer_scores.append(score)
    max_layer_score = max(layer_scores) if layer_scores and max(layer_scores) > 0 else 1

    for sc_name, score in zip(layer_scenarios, layer_scores):
        n_val = sc_name.replace("layers_", "")
        html += (
            f'<tr><td>{n_val}</td>'
            f'<td class="num">{_format_number(score)}</td>'
            f'<td>{_bar_html(score, max_layer_score, _format_number(score), "bar-purple")}</td></tr>\n'
        )
    html += '</table></div>\n'

    return html


def section_best_config(s3_base: str) -> str:
    """Section 6: Best Config Full Pipeline Results."""
    html = '<h1>6. Best Config Full Pipeline</h1>\n'

    # Teacher training curve
    train_metrics = collect_training_metrics(s3_base, "phase4", "teacher")
    html += '<div class="card"><div class="card-header">Teacher Training Curve</div>\n'
    if train_metrics:
        epochs = train_metrics.get("epoch_history", [])
        html += '<table><tr><th>Epoch</th><th>Train Loss</th><th>Val Loss</th><th>Val AUC</th></tr>\n'
        for ep in epochs:
            html += (
                f'<tr><td class="num">{ep.get("epoch", "-")}</td>'
                f'<td class="num">{_format_number(ep.get("train_loss"), 4)}</td>'
                f'<td class="num">{_format_number(ep.get("val_loss"), 4)}</td>'
                f'<td class="num">{_format_number(ep.get("val_auc"), 4)}</td></tr>\n'
            )
        html += '</table>\n'
    else:
        html += '<p>Training metrics not available.</p>\n'
    html += '</div>\n'

    # Distillation results (16 students)
    distill_summary = _download_json(
        f"{s3_base}/phase4/distillation/output/distillation_summary.json"
    )
    html += '<div class="card"><div class="card-header">Distillation Results (16 Students)</div>\n'
    if distill_summary:
        html += f'<p>Temperature: {distill_summary.get("temperature", "?")}, '
        html += f'Alpha: {distill_summary.get("alpha", "?")}, '
        html += f'Features: {distill_summary.get("feature_count", "?")}</p>\n'

        per_task = distill_summary.get("fidelity", {}).get("per_task", {})
        html += '<table><tr><th>Task</th><th>Passed</th><th>Metrics</th></tr>\n'
        for task_name, info in per_task.items():
            passed = info.get("passed", False)
            badge = "badge-pass" if passed else "badge-fail"
            metrics_str = ", ".join(
                f"{k}={_format_number(v)}"
                for k, v in info.get("metrics", {}).items()
            )
            html += (
                f'<tr><td>{_html_escape(task_name)}</td>'
                f'<td><span class="badge {badge}">{"PASS" if passed else "FAIL"}</span></td>'
                f'<td>{_html_escape(metrics_str)}</td></tr>\n'
            )
        html += '</table>\n'
    else:
        html += '<p>Distillation summary not available.</p>\n'
    html += '</div>\n'

    # Fidelity matrix
    fidelity_report = _download_json(
        f"{s3_base}/phase4/distillation/output/fidelity_report.json"
    )
    html += '<div class="card"><div class="card-header">Fidelity Validation (8 Metrics x 16 Tasks)</div>\n'
    if fidelity_report:
        details = fidelity_report.get("details", {})
        # Collect metric names from first task
        metric_names = []
        for task_info in details.values():
            metric_names = list(task_info.get("metrics", {}).keys())
            break

        if metric_names:
            html += '<div style="overflow-x: auto;">\n'
            html += '<table><tr><th>Task</th>'
            for mn in metric_names:
                html += f'<th>{_html_escape(mn)}</th>'
            html += '<th>Status</th></tr>\n'
            for task_name, info in details.items():
                passed = info.get("passed", False)
                badge = "badge-pass" if passed else "badge-fail"
                html += f'<tr><td>{_html_escape(task_name)}</td>'
                for mn in metric_names:
                    val = info.get("metrics", {}).get(mn)
                    html += f'<td class="num">{_format_number(val)}</td>'
                html += f'<td><span class="badge {badge}">{"PASS" if passed else "FAIL"}</span></td></tr>\n'
            html += '</table></div>\n'
    else:
        html += '<p>Fidelity report not available.</p>\n'
    html += '</div>\n'

    # Feature selection
    if distill_summary:
        fs = distill_summary.get("feature_selection", {})
        html += '<div class="card"><div class="card-header">Feature Selection Per Task</div>\n'
        if fs:
            html += '<table><tr><th>Task</th><th>Selected</th><th>Original</th><th>Reduction</th><th>Selection Ratio</th></tr>\n'
            for task_name, info in fs.items():
                selected = info.get("selected", 0)
                original = info.get("original", 0)
                reduction = info.get("reduction_pct", 0)
                ratio = selected / original if original > 0 else 0
                html += (
                    f'<tr><td>{_html_escape(task_name)}</td>'
                    f'<td class="num">{selected}</td>'
                    f'<td class="num">{original}</td>'
                    f'<td class="num">{reduction:.1f}%</td>'
                    f'<td>{_bar_html(ratio, 1.0, f"{ratio:.0%}", "bar-cyan")}</td></tr>\n'
                )
            html += '</table>\n'
        html += '</div>\n'

    return html


def section_per_task_details(s3_base: str) -> str:
    """Section 7: Per-Task Detailed Results."""
    html = '<h1>7. Per-Task Details</h1>\n'

    eval_metrics = _download_json(
        f"{s3_base}/phase4/distillation/output/eval_metrics.json"
    ) or _download_json(
        f"{s3_base}/phase4/teacher/output/eval_metrics.json"
    )

    for task_name in TASK_NAMES:
        task_type = TASK_TYPES[task_name]

        html += f'<details><summary>{_html_escape(task_name)} ({task_type})</summary><div>\n'

        if eval_metrics:
            task_data = eval_metrics.get("per_task", {}).get(task_name, {})
            if task_data:
                html += '<table><tr><th>Metric</th><th>Value</th></tr>\n'
                for metric_name, metric_val in task_data.items():
                    if isinstance(metric_val, (int, float)):
                        html += (
                            f'<tr><td>{_html_escape(metric_name)}</td>'
                            f'<td class="num">{_format_number(metric_val)}</td></tr>\n'
                        )
                html += '</table>\n'

                # Type-specific visualizations
                if task_type == "binary":
                    # ROC/PR curve data points
                    roc_data = task_data.get("roc_curve", {})
                    if roc_data:
                        html += '<h3>ROC Curve Data</h3>\n'
                        html += '<table><tr><th>FPR</th><th>TPR</th></tr>\n'
                        fprs = roc_data.get("fpr", [])
                        tprs = roc_data.get("tpr", [])
                        # Sample every Nth point for display
                        step = max(1, len(fprs) // 10)
                        for i in range(0, len(fprs), step):
                            html += f'<tr><td class="num">{_format_number(fprs[i])}</td><td class="num">{_format_number(tprs[i])}</td></tr>\n'
                        html += '</table>\n'

                    pr_data = task_data.get("pr_curve", {})
                    if pr_data:
                        html += '<h3>Precision-Recall Curve Data</h3>\n'
                        html += '<table><tr><th>Recall</th><th>Precision</th></tr>\n'
                        recalls = pr_data.get("recall", [])
                        precisions = pr_data.get("precision", [])
                        step = max(1, len(recalls) // 10)
                        for i in range(0, len(recalls), step):
                            html += f'<tr><td class="num">{_format_number(recalls[i])}</td><td class="num">{_format_number(precisions[i])}</td></tr>\n'
                        html += '</table>\n'

                elif task_type == "multiclass":
                    # Confusion matrix
                    cm = task_data.get("confusion_matrix")
                    if cm and isinstance(cm, list):
                        html += '<h3>Confusion Matrix</h3>\n'
                        n_classes = len(cm)
                        html += '<table><tr><th></th>'
                        for c in range(n_classes):
                            html += f'<th>Pred {c}</th>'
                        html += '</tr>\n'
                        for r in range(n_classes):
                            html += f'<tr><th>True {r}</th>'
                            row_sum = sum(cm[r]) if isinstance(cm[r], list) else 1
                            for c in range(n_classes):
                                val = cm[r][c] if isinstance(cm[r], list) else 0
                                intensity = val / row_sum if row_sum > 0 else 0
                                bg = f"rgba(88,166,255,{intensity:.2f})"
                                html += f'<td class="num" style="background:{bg}">{val}</td>'
                            html += '</tr>\n'
                        html += '</table>\n'

                elif task_type == "regression":
                    # Residual stats
                    residuals = task_data.get("residual_stats", {})
                    if residuals:
                        html += '<h3>Residual Statistics</h3>\n'
                        html += '<table><tr><th>Stat</th><th>Value</th></tr>\n'
                        for k, v in residuals.items():
                            html += f'<tr><td>{_html_escape(k)}</td><td class="num">{_format_number(v)}</td></tr>\n'
                        html += '</table>\n'
            else:
                html += '<p>No evaluation data for this task.</p>\n'
        else:
            html += '<p>Evaluation metrics not available.</p>\n'

        html += '</div></details>\n'

    return html


def section_reason_validation(s3_base: str) -> str:
    """Section 8: Reason Generation Validation."""
    html = '<h1>8. Reason Generation Validation</h1>\n'

    reason_data = _download_json(
        f"{s3_base}/phase4/distillation/output/reason_samples.json"
    )

    html += '<div class="card"><div class="card-header">L1 Template Reason Samples</div>\n'
    if reason_data:
        samples = reason_data.get("samples", [])
        for sample in samples[:5]:
            task = sample.get("task", "?")
            reason = sample.get("reason", "")
            features = sample.get("top_features", [])
            html += f'<div style="margin: 8px 0; padding: 8px; background: var(--bg-tertiary); border-radius: 4px;">\n'
            html += f'<strong>{_html_escape(task)}</strong><br>\n'
            html += f'<span style="color: var(--accent-green);">{_html_escape(reason)}</span><br>\n'
            if features:
                html += f'<span style="color: var(--text-muted); font-size: 0.85em;">Top features: {_html_escape(", ".join(str(f) for f in features[:5]))}</span>\n'
            html += '</div>\n'
    else:
        html += '<p>Reason generation samples not available.</p>\n'
    html += '</div>\n'

    # Self-critique results
    critique_data = _download_json(
        f"{s3_base}/phase4/distillation/output/self_critique.json"
    )
    html += '<div class="card"><div class="card-header">Self-Critique Results</div>\n'
    if critique_data:
        html += '<table><tr><th>Task</th><th>Score</th><th>Issues</th></tr>\n'
        for task_name, info in critique_data.items():
            score = info.get("score", 0)
            issues = info.get("issues", [])
            html += (
                f'<tr><td>{_html_escape(task_name)}</td>'
                f'<td class="num">{_format_number(score, 2)}</td>'
                f'<td>{_html_escape("; ".join(issues) if issues else "None")}</td></tr>\n'
            )
        html += '</table>\n'
    else:
        html += '<p>Self-critique results not available.</p>\n'
    html += '</div>\n'

    # IG feature interpretation
    ig_data = _download_json(
        f"{s3_base}/phase4/distillation/output/ig_interpretation.json"
    )
    html += '<div class="card"><div class="card-header">Integrated Gradients Feature Interpretation</div>\n'
    if ig_data:
        for task_name, features in list(ig_data.items())[:5]:
            html += f'<h3>{_html_escape(task_name)}</h3>\n'
            if isinstance(features, dict):
                sorted_feats = sorted(features.items(), key=lambda x: abs(float(x[1])), reverse=True)[:10]
                max_attr = max(abs(float(v)) for _, v in sorted_feats) if sorted_feats else 1
                for feat_name, attr_val in sorted_feats:
                    attr_f = float(attr_val)
                    color = "bar-green" if attr_f >= 0 else "bar-red"
                    html += (
                        f'<div class="chart-row">'
                        f'<span class="chart-label">{_html_escape(feat_name[:20])}</span>'
                        f'<div class="chart-bar-wrap">'
                        f'<div class="chart-bar {color}" style="width:{abs(attr_f)/max_attr*100:.1f}%">'
                        f'{attr_f:+.4f}</div></div></div>\n'
                    )
    else:
        html += '<p>IG interpretation data not available.</p>\n'
    html += '</div>\n'

    return html


def section_pipeline_details(s3_base: str, manifest: Optional[dict]) -> str:
    """Section 9: Pipeline Details (timeline, jobs, hyperparameters)."""
    html = '<h1>9. Pipeline Details</h1>\n'

    # Timeline
    html += '<div class="card"><div class="card-header">Execution Timeline</div>\n'
    html += '<div class="timeline">\n'

    phase_names = {
        "phase0": "Data Preparation",
        "phase1": "Feature Group Ablation (9 scenarios)",
        "phase2": "Expert Ablation (7 scenarios)",
        "phase3": "Hyperparameter Sensitivity (12 scenarios)",
        "phase4": "Best Config Full Pipeline",
        "phase5": "Result Collection + Report",
    }

    phases = manifest.get("phases", {}) if manifest else {}
    for phase_key, phase_label in phase_names.items():
        phase_data = phases.get(phase_key)
        if phase_data is None:
            status_cls = ""
            status_badge = '<span class="badge badge-warn">SKIPPED</span>'
        elif isinstance(phase_data, list):
            completed = sum(1 for r in phase_data if r.get("status") == "Completed")
            failed = sum(1 for r in phase_data if r.get("status") == "Failed")
            total = len(phase_data)
            if failed > 0:
                status_cls = "failed"
                status_badge = f'<span class="badge badge-fail">{completed}/{total} OK, {failed} FAILED</span>'
            elif completed == total:
                status_cls = "completed"
                status_badge = f'<span class="badge badge-pass">{total}/{total} COMPLETED</span>'
            else:
                status_cls = ""
                status_badge = f'<span class="badge badge-info">{completed}/{total} COMPLETED</span>'
        elif isinstance(phase_data, dict):
            st = phase_data.get("status", "Unknown")
            status_cls = "completed" if st == "Completed" else ("failed" if st == "Failed" else "")
            badge_cls = "badge-pass" if st == "Completed" else ("badge-fail" if st == "Failed" else "badge-info")
            status_badge = f'<span class="badge {badge_cls}">{st}</span>'
        else:
            status_cls = ""
            status_badge = '<span class="badge badge-info">?</span>'

        html += (
            f'<div class="timeline-item {status_cls}">'
            f'<strong>{_html_escape(phase_key)}</strong>: {_html_escape(phase_label)} '
            f'{status_badge}</div>\n'
        )
    html += '</div></div>\n'

    # Job listing
    html += '<div class="card"><div class="card-header">SageMaker Job List</div>\n'
    html += '<table><tr><th>Phase</th><th>Job Name</th><th>Status</th><th>Scenario</th></tr>\n'

    for phase_key in phase_names.keys():
        phase_data = phases.get(phase_key)
        if phase_data is None:
            continue
        if isinstance(phase_data, list):
            for item in phase_data:
                job_name = item.get("job_name", "-")
                status = item.get("status", "?")
                scenario = item.get("scenario", "-")
                badge_cls = (
                    "badge-pass" if status == "Completed"
                    else "badge-fail" if status == "Failed"
                    else "badge-info"
                )
                html += (
                    f'<tr><td>{phase_key}</td>'
                    f'<td><code>{_html_escape(job_name)}</code></td>'
                    f'<td><span class="badge {badge_cls}">{status}</span></td>'
                    f'<td>{_html_escape(scenario)}</td></tr>\n'
                )
        elif isinstance(phase_data, dict):
            job_name = phase_data.get("job_name", "-")
            status = phase_data.get("status", "?")
            badge_cls = (
                "badge-pass" if status == "Completed"
                else "badge-fail" if status == "Failed"
                else "badge-info"
            )
            html += (
                f'<tr><td>{phase_key}</td>'
                f'<td><code>{_html_escape(job_name)}</code></td>'
                f'<td><span class="badge {badge_cls}">{status}</span></td>'
                f'<td>-</td></tr>\n'
            )
    html += '</table></div>\n'

    # Hyperparameter record
    html += '<div class="card"><div class="card-header">Hyperparameter Record</div>\n'
    html += '<details><summary>Feature Group Ablation Scenarios</summary><div>\n'
    html += '<table><tr><th>Scenario</th><th>Removed Groups</th></tr>\n'
    from scripts.run_ablation_test import FEATURE_ABLATION_SCENARIOS
    for sc in FEATURE_ABLATION_SCENARIOS:
        html += f'<tr><td>{_html_escape(sc["name"])}</td><td>{_html_escape(", ".join(sc["remove"]) or "none")}</td></tr>\n'
    html += '</table></div></details>\n'

    html += '<details><summary>Expert Ablation Scenarios</summary><div>\n'
    html += '<table><tr><th>Scenario</th><th>Shared Experts</th></tr>\n'
    from scripts.run_ablation_test import EXPERT_ABLATION_SCENARIOS
    for sc in EXPERT_ABLATION_SCENARIOS:
        html += f'<tr><td>{_html_escape(sc["name"])}</td><td>{_html_escape(", ".join(sc["shared"]))}</td></tr>\n'
    html += '</table></div></details>\n'

    html += '<details><summary>Hyperparameter Sensitivity Scenarios</summary><div>\n'
    html += '<table><tr><th>Scenario</th><th>Parameter</th><th>Value</th></tr>\n'
    from scripts.run_ablation_test import HP_SCENARIOS
    for sc in HP_SCENARIOS:
        name = sc["name"]
        if "learning_rate" in sc:
            html += f'<tr><td>{name}</td><td>learning_rate</td><td>{sc["learning_rate"]}</td></tr>\n'
        elif "temperature" in sc:
            html += f'<tr><td>{name}</td><td>temperature</td><td>{sc["temperature"]}</td></tr>\n'
        elif "num_layers" in sc:
            html += f'<tr><td>{name}</td><td>num_layers</td><td>{sc["num_layers"]}</td></tr>\n'
    html += '</table></div></details>\n'
    html += '</div>\n'

    return html


# ===================================================================
# Report assembly
# ===================================================================

def generate_report(s3_base: str, output_path: str) -> str:
    """Generate the full ablation HTML report.

    Parameters
    ----------
    s3_base : str
        S3 base path for the ablation test run
        (e.g. s3://aiops-ple-financial/ablation-test/20260320-143000).
    output_path : str
        Local path for the output HTML file.

    Returns
    -------
    str
        The output file path.
    """
    logger.info("Generating ablation report from %s", s3_base)
    manifest = collect_manifest(s3_base)

    ts = s3_base.rstrip("/").split("/")[-1]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build sections
    sections = []
    sections.append(section_overview(manifest, s3_base))
    sections.append(section_data_quality(s3_base))
    sections.append(section_feature_ablation(s3_base))
    sections.append(section_expert_ablation(s3_base))
    sections.append(section_hp_sensitivity(s3_base))
    sections.append(section_best_config(s3_base))
    sections.append(section_per_task_details(s3_base))
    sections.append(section_reason_validation(s3_base))

    try:
        sections.append(section_pipeline_details(s3_base, manifest))
    except ImportError:
        logger.warning("Could not import scenario definitions for pipeline details section")

    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Ablation Test Report — {_html_escape(ts)}</title>
    <style>{DARK_CSS}</style>
</head>
<body>
    <header style="text-align: center; margin-bottom: 24px;">
        <h1 style="border: none; font-size: 2em; margin-bottom: 4px;">
            Ablation Test Report
        </h1>
        <p style="color: var(--text-muted);">
            ealtman2019 Credit Card Transactions | {_html_escape(ts)} | Generated {_html_escape(now)}
        </p>
    </header>

    {body}

    <div class="footer">
        Generated by <code>scripts/generate_ablation_report.py</code>
        | AIOps PLE Financial Pipeline
        | {_html_escape(now)}
    </div>
</body>
</html>"""

    # Write output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    file_size = os.path.getsize(output_path) / 1024
    logger.info("Report written to %s (%.1f KB)", output_path, file_size)
    return output_path


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HTML report from ablation test results on S3",
    )
    parser.add_argument(
        "--s3-base",
        type=str,
        required=True,
        help="S3 base path (e.g. s3://aiops-ple-financial/ablation-test/20260320-143000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output HTML path (default: docs/ablation_report_{timestamp}.html)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    s3_base = args.s3_base.rstrip("/")

    ts = s3_base.split("/")[-1]
    output_path = args.output or f"docs/ablation_report_{ts}.html"

    generate_report(s3_base, output_path)


if __name__ == "__main__":
    main()
