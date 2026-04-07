"""HTML usage report generator for token/cost tracking."""

import html
import os
import tempfile
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .cost_tracker import CostTracker, APICall


def _fmt_tokens(n: int) -> str:
    """Format token count with commas."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:,}"


def _fmt_cost(c: float) -> str:
    """Format cost as dollar amount."""
    if c < 0.01:
        return f"${c:.4f}"
    return f"${c:.2f}"


def _fmt_duration(ms: float) -> str:
    """Format duration in human-friendly form."""
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m = int(s) // 60
    sec = int(s) % 60
    return f"{m}m {sec}s"


def _time_ago(dt: datetime) -> str:
    """Format a datetime as relative time."""
    delta = datetime.now() - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        m = secs // 60
        return f"{m}m ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h}h ago"
    d = secs // 86400
    return f"{d}d ago"


def generate_usage_html(
    tracker: CostTracker,
    provider_name: str = "",
    session_name: str = "",
) -> str:
    """Generate a full HTML usage report from the cost tracker."""
    summary = tracker.get_summary()
    by_model = tracker.get_cost_by_model()
    calls = tracker.calls

    # Build timeline buckets (per-minute)
    timeline: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for call in calls:
        bucket = call.timestamp.strftime("%H:%M")
        timeline[bucket]["input"] += call.input_tokens
        timeline[bucket]["output"] += call.output_tokens

    # Sort timeline keys
    sorted_times = sorted(timeline.keys())
    timeline_labels = sorted_times or ["now"]
    timeline_input = [timeline[t]["input"] for t in sorted_times] or [0]
    timeline_output = [timeline[t]["output"] for t in sorted_times] or [0]

    # Per-model rows
    model_rows_html = ""
    sorted_models = sorted(
        by_model.items(), key=lambda kv: kv[1]["total_tokens"], reverse=True
    )
    total_tokens_all = max(summary.total_tokens, 1)
    for model, row in sorted_models:
        pct = (row["total_tokens"] / total_tokens_all) * 100
        model_rows_html += f"""
        <tr>
          <td class="model-name">{html.escape(model)}</td>
          <td class="num">{int(row['calls'])}</td>
          <td class="num">{_fmt_tokens(int(row['input_tokens']))}</td>
          <td class="num">{_fmt_tokens(int(row['output_tokens']))}</td>
          <td class="num bold">{_fmt_tokens(int(row['total_tokens']))}</td>
          <td class="num">{_fmt_cost(row['total_cost'])}</td>
          <td>
            <div class="bar-bg"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
          </td>
        </tr>"""

    # Recent calls table (last 30)
    recent_html = ""
    for call in reversed(calls[-30:]):
        reason_badge = ""
        if call.finish_reason and call.finish_reason != "stop":
            cls = "badge-warn" if call.finish_reason in ("length", "interrupted") else "badge-info"
            reason_badge = f'<span class="badge {cls}">{html.escape(call.finish_reason)}</span>'
        recent_html += f"""
        <tr>
          <td class="dim">{call.timestamp.strftime('%H:%M:%S')}</td>
          <td class="model-name">{html.escape(call.model)}</td>
          <td class="num">{_fmt_tokens(call.input_tokens)}</td>
          <td class="num">{_fmt_tokens(call.output_tokens)}</td>
          <td class="num">{_fmt_cost(call.total_cost)}</td>
          <td class="num">{call.tool_calls}</td>
          <td>{reason_badge}</td>
        </tr>"""

    # Extra usage (cache, reasoning tokens, etc.)
    extra_html = ""
    if summary.extra_usage_totals:
        extra_items = []
        for key in (
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "prompt_cached_tokens",
            "completion_reasoning_tokens",
            "reasoning_tokens",
        ):
            if key in summary.extra_usage_totals:
                label = key.replace("_", " ").replace("tokens", "").strip().title()
                val = summary.extra_usage_totals[key]
                extra_items.append(f"""
                <div class="stat-card mini">
                  <div class="stat-label">{html.escape(label)}</div>
                  <div class="stat-value">{_fmt_tokens(val)}</div>
                </div>""")
        if extra_items:
            extra_html = f"""
            <section>
              <h2>Extended Token Metrics</h2>
              <div class="stats-row">{''.join(extra_items)}</div>
            </section>"""

    session_start = tracker.session_start.strftime("%Y-%m-%d %H:%M")
    elapsed = datetime.now() - tracker.session_start
    elapsed_str = _fmt_duration(elapsed.total_seconds() * 1000)
    subtitle_parts = []
    if provider_name:
        subtitle_parts.append(provider_name)
    if session_name:
        subtitle_parts.append(f"session: {session_name}")
    subtitle_parts.append(f"started {session_start}")
    subtitle_parts.append(f"elapsed {elapsed_str}")
    subtitle = " · ".join(subtitle_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Usage Report — harness</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --accent3: #d2a8ff;
    --warn: #d29922;
    --red: #f85149;
    --input-color: #58a6ff;
    --output-color: #3fb950;
    --radius: 10px;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  header {{
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    font-size: 1.6rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: .5rem;
  }}
  header h1 .icon {{ font-size: 1.3rem; }}
  .subtitle {{ color: var(--text-dim); font-size: .85rem; margin-top: .35rem; }}

  /* Stat cards */
  .stats-row {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 150px;
  }}
  .stat-card.mini {{
    padding: .8rem 1rem;
    min-width: 120px;
  }}
  .stat-label {{
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--text-dim);
    margin-bottom: .3rem;
  }}
  .stat-value {{
    font-size: 1.5rem;
    font-weight: 700;
  }}
  .stat-card.mini .stat-value {{ font-size: 1.1rem; }}
  .stat-sub {{ font-size: .8rem; color: var(--text-dim); margin-top: .15rem; }}
  .color-input {{ color: var(--input-color); }}
  .color-output {{ color: var(--output-color); }}
  .color-accent {{ color: var(--accent3); }}

  /* Sections */
  section {{ margin-bottom: 2.5rem; }}
  section h2 {{
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: .05em;
  }}

  /* Tables */
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: .85rem;
  }}
  thead th {{
    text-align: left;
    padding: .7rem 1rem;
    font-weight: 600;
    font-size: .75rem;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }}
  thead th.num {{ text-align: right; }}
  tbody td {{
    padding: .6rem 1rem;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
  }}
  tbody tr:last-child td {{ border-bottom: none; }}
  tbody tr:hover {{ background: rgba(88,166,255,.05); }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.bold {{ font-weight: 600; }}
  td.dim {{ color: var(--text-dim); }}
  td.model-name {{
    font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    font-size: .82rem;
  }}

  /* Bar chart in table */
  .bar-bg {{
    height: 8px;
    background: var(--border);
    border-radius: 4px;
    min-width: 80px;
    overflow: hidden;
  }}
  .bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, var(--input-color), var(--accent3));
    border-radius: 4px;
    transition: width .3s ease;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    padding: .15rem .5rem;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 500;
  }}
  .badge-warn {{ background: rgba(210,153,34,.15); color: var(--warn); }}
  .badge-info {{ background: rgba(88,166,255,.15); color: var(--accent); }}

  /* Chart */
  .chart-container {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
  }}
  canvas {{ width: 100% !important; height: 220px !important; }}

  /* Footer */
  footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center;
    color: var(--text-dim);
    font-size: .75rem;
  }}

  @media (max-width: 700px) {{
    body {{ padding: 1rem; }}
    .stats-row {{ flex-direction: column; }}
    .stat-card {{ min-width: unset; }}
  }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1><span class="icon">📊</span> Usage Report</h1>
    <div class="subtitle">{html.escape(subtitle)}</div>
  </header>

  <!-- Summary cards -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">Total Tokens</div>
      <div class="stat-value">{_fmt_tokens(summary.total_tokens)}</div>
      <div class="stat-sub">{summary.total_calls} API call{"s" if summary.total_calls != 1 else ""}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Input Tokens</div>
      <div class="stat-value color-input">{_fmt_tokens(summary.total_input_tokens)}</div>
      <div class="stat-sub">{_fmt_cost(summary.total_input_cost)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Output Tokens</div>
      <div class="stat-value color-output">{_fmt_tokens(summary.total_output_tokens)}</div>
      <div class="stat-sub">{_fmt_cost(summary.total_output_cost)}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Cost</div>
      <div class="stat-value color-accent">{_fmt_cost(summary.total_cost)}</div>
      <div class="stat-sub">{_fmt_cost(summary.total_cost / max(1, summary.total_calls))} avg/call</div>
    </div>
  </div>

  {extra_html}

  <!-- Timeline chart -->
  {"" if len(sorted_times) < 2 else f'''
  <section>
    <h2>Token Usage Over Time</h2>
    <div class="chart-container">
      <canvas id="timelineChart"></canvas>
    </div>
  </section>
  '''}

  <!-- Per-model breakdown -->
  <section>
    <h2>Usage by Model</h2>
    {"<p style='color:var(--text-dim);font-size:.85rem'>No API calls recorded yet.</p>" if not by_model else f'''
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th class="num">Calls</th>
          <th class="num">Input</th>
          <th class="num">Output</th>
          <th class="num">Total</th>
          <th class="num">Cost</th>
          <th>Share</th>
        </tr>
      </thead>
      <tbody>{model_rows_html}</tbody>
    </table>
    '''}
  </section>

  <!-- Recent calls -->
  {"" if not calls else f'''
  <section>
    <h2>Recent API Calls (last {min(len(calls), 30)})</h2>
    <table>
      <thead>
        <tr>
          <th>Time</th>
          <th>Model</th>
          <th class="num">Input</th>
          <th class="num">Output</th>
          <th class="num">Cost</th>
          <th class="num">Tools</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>{recent_html}</tbody>
    </table>
  </section>
  '''}

  <footer>
    Generated by harness · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </footer>
</div>

{"" if len(sorted_times) < 2 else f'''
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script>
const ctx = document.getElementById("timelineChart").getContext("2d");
new Chart(ctx, {{
  type: "bar",
  data: {{
    labels: {timeline_labels},
    datasets: [
      {{
        label: "Input Tokens",
        data: {timeline_input},
        backgroundColor: "rgba(88,166,255,.6)",
        borderRadius: 4,
      }},
      {{
        label: "Output Tokens",
        data: {timeline_output},
        backgroundColor: "rgba(63,185,80,.6)",
        borderRadius: 4,
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{
        labels: {{ color: "#8b949e", font: {{ size: 12 }} }}
      }},
      tooltip: {{
        backgroundColor: "#161b22",
        titleColor: "#e6edf3",
        bodyColor: "#e6edf3",
        borderColor: "#30363d",
        borderWidth: 1,
        callbacks: {{
          label: (ctx) => ctx.dataset.label + ": " + ctx.raw.toLocaleString() + " tokens"
        }}
      }}
    }},
    scales: {{
      x: {{
        stacked: true,
        grid: {{ color: "rgba(48,54,61,.4)" }},
        ticks: {{ color: "#8b949e", font: {{ size: 11 }} }}
      }},
      y: {{
        stacked: true,
        grid: {{ color: "rgba(48,54,61,.4)" }},
        ticks: {{
          color: "#8b949e",
          font: {{ size: 11 }},
          callback: (v) => v >= 1000000 ? (v/1000000).toFixed(1)+"M" : v >= 1000 ? (v/1000).toFixed(0)+"k" : v
        }}
      }}
    }}
  }}
}});
</script>
'''}
</body>
</html>"""


def open_usage_report(
    tracker: CostTracker,
    provider_name: str = "",
    session_name: str = "",
) -> str:
    """Generate the HTML report and open it in the default browser.

    Returns the path to the generated HTML file.
    """
    html_content = generate_usage_html(tracker, provider_name, session_name)

    report_dir = os.path.join(tempfile.gettempdir(), "harness_reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "usage_report.html")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    webbrowser.open(f"file:///{report_path.replace(os.sep, '/')}")
    return report_path
