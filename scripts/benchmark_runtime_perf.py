"""
Benchmark runtime performance presets using dashboard_state telemetry.

Measures:
- CPU %
- RAM %
- FPS
- Detection latency (inference avg ms)
- Frame processing latency (avg ms)

Outputs:
- JSON report
- Markdown report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PRESET_PATCHES: Dict[str, Dict[str, Any]] = {
    "eco": {
        "performance_preset": "eco",
        "target_fps": 6,
        "inference_mode": "motion_gated",
        "dashboard_ui_frame_interval_sec": 0.35,
        "dashboard_state_min_interval_sec": 1.5,
        "gc_collect_interval_sec": 120.0,
        "summary_log_interval_sec": 45.0,
        "auto_motion_gate_enabled": True,
        "auto_motion_gate_cpu_threshold": 70.0,
        "auto_motion_gate_mem_threshold": 78.0,
        "auto_motion_gate_recover_sec": 30.0,
        "auto_motion_gate_min_hold_sec": 20.0,
    },
    "balanced": {
        "performance_preset": "balanced",
        "target_fps": 10,
        "inference_mode": "always",
        "dashboard_ui_frame_interval_sec": 0.20,
        "dashboard_state_min_interval_sec": 1.0,
        "gc_collect_interval_sec": 60.0,
        "summary_log_interval_sec": 30.0,
        "auto_motion_gate_enabled": True,
        "auto_motion_gate_cpu_threshold": 85.0,
        "auto_motion_gate_mem_threshold": 88.0,
        "auto_motion_gate_recover_sec": 20.0,
        "auto_motion_gate_min_hold_sec": 15.0,
    },
    "max_accuracy": {
        "performance_preset": "max_accuracy",
        "target_fps": 15,
        "inference_mode": "always",
        "dashboard_ui_frame_interval_sec": 0.12,
        "dashboard_state_min_interval_sec": 0.60,
        "gc_collect_interval_sec": 45.0,
        "summary_log_interval_sec": 20.0,
        "auto_motion_gate_enabled": True,
        "auto_motion_gate_cpu_threshold": 92.0,
        "auto_motion_gate_mem_threshold": 94.0,
        "auto_motion_gate_recover_sec": 10.0,
        "auto_motion_gate_min_hold_sec": 8.0,
    },
}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _apply_override_patch(override_path: Path, patch: Dict[str, Any]) -> None:
    current = _read_json(override_path)
    current.update(patch)
    _write_json(override_path, current)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * max(0.0, min(1.0, p))
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(ordered[lo])
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "avg": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "p95": float(_percentile(values, 0.95)),
    }


def _extract_sample(state: Dict[str, Any], max_stale_sec: float = 5.0) -> Optional[Dict[str, float]]:
    ts = float(state.get("timestamp", 0.0) or 0.0)
    if ts <= 0.0:
        return None
    age = time.time() - ts
    if age > max_stale_sec:
        return None

    status = state.get("status", {}) if isinstance(state.get("status"), dict) else {}
    resources = status.get("resources", {}) if isinstance(status.get("resources"), dict) else {}
    perf = status.get("perf", {}) if isinstance(status.get("perf"), dict) else {}
    return {
        "cpu_percent": float(resources.get("cpu_percent", 0.0) or 0.0),
        "memory_percent": float(resources.get("memory_percent", 0.0) or 0.0),
        "fps": float(resources.get("fps", 0.0) or 0.0),
        "inference_latency_ms": float(perf.get("inference_latency_ms_avg", 0.0) or 0.0),
        "frame_latency_ms": float(perf.get("frame_process_latency_ms_avg", 0.0) or 0.0),
    }


def _collect_samples(
    state_file: Path,
    duration_sec: float,
    poll_sec: float,
) -> List[Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    end_at = time.time() + max(1.0, duration_sec)
    while time.time() < end_at:
        sample = _extract_sample(_read_json(state_file))
        if sample is not None:
            samples.append(sample)
        time.sleep(max(0.1, poll_sec))
    return samples


def _wait_for_fresh_state(state_file: Path, timeout_sec: float) -> bool:
    end_at = time.time() + max(1.0, timeout_sec)
    while time.time() < end_at:
        sample = _extract_sample(_read_json(state_file), max_stale_sec=15.0)
        if sample is not None:
            return True
        time.sleep(0.4)
    return False


def _summarize_samples(samples: List[Dict[str, float]]) -> Dict[str, Any]:
    cpu = [s["cpu_percent"] for s in samples]
    mem = [s["memory_percent"] for s in samples]
    fps = [s["fps"] for s in samples]
    infer = [s["inference_latency_ms"] for s in samples]
    frame = [s["frame_latency_ms"] for s in samples]
    return {
        "samples": len(samples),
        "cpu_percent": _stat(cpu),
        "memory_percent": _stat(mem),
        "fps": _stat(fps),
        "inference_latency_ms": _stat(infer),
        "frame_latency_ms": _stat(frame),
    }


def _python_has_numpy(py_exe: Path) -> bool:
    try:
        proc = subprocess.run(
            [str(py_exe), "-c", "import numpy"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _resolve_runtime_python(root: Path, preferred: str = "") -> Path:
    if preferred:
        candidate = Path(preferred).expanduser().resolve()
        if candidate.exists():
            return candidate
    candidates = [Path(sys.executable)]
    if os.name == "nt":
        candidates.extend([
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "Scripts" / "python.exe",
        ])
    else:
        candidates.extend([
            root / ".venv" / "bin" / "python",
            root / "venv" / "bin" / "python",
        ])
    for py in candidates:
        if py.exists() and _python_has_numpy(py):
            return py
    return Path(sys.executable)


def _start_runtime(root: Path, log_file: Path, runtime_python: Path) -> subprocess.Popen:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = log_file.open("a", encoding="utf-8")
    cmd = [str(runtime_python), "-u", str(root / "runtime" / "agent_v2.py")]
    proc = subprocess.Popen(cmd, cwd=str(root), stdout=fh, stderr=subprocess.STDOUT)
    setattr(proc, "_benchmark_log_fh", fh)
    return proc


def _stop_runtime(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        fh_done = getattr(proc, "_benchmark_log_fh", None)
        if fh_done is not None:
            try:
                fh_done.close()
            except Exception:
                pass
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    fh = getattr(proc, "_benchmark_log_fh", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _delta_pct(base: float, value: float) -> float:
    if abs(base) < 1e-9:
        return 0.0
    return ((value - base) / base) * 100.0


def _render_markdown(
    report: Dict[str, Any],
    preset_order: List[str],
) -> str:
    lines: List[str] = []
    lines.append("# Runtime Performance Benchmark")
    lines.append("")
    lines.append(f"- Generated at: `{report.get('generated_at')}`")
    lines.append(f"- Sample seconds/preset: `{report.get('sample_sec')}`")
    lines.append(f"- Warmup seconds/preset: `{report.get('warmup_sec')}`")
    lines.append(f"- Poll interval: `{report.get('poll_sec')}`")
    lines.append("")
    lines.append("## Metrics (Average)")
    lines.append("")
    lines.append("| Preset | Samples | CPU % | RAM % | FPS | Detection Latency ms | Frame Latency ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for preset in preset_order:
        item = report["results"].get(preset, {})
        lines.append(
            f"| {preset} | {int(item.get('samples', 0))} | "
            f"{item.get('cpu_percent', {}).get('avg', 0.0):.2f} | "
            f"{item.get('memory_percent', {}).get('avg', 0.0):.2f} | "
            f"{item.get('fps', {}).get('avg', 0.0):.2f} | "
            f"{item.get('inference_latency_ms', {}).get('avg', 0.0):.2f} | "
            f"{item.get('frame_latency_ms', {}).get('avg', 0.0):.2f} |"
        )

    base = preset_order[0] if preset_order else ""
    if base and base in report["results"]:
        lines.append("")
        lines.append(f"## Delta Vs `{base}` (Average, %)")
        lines.append("")
        lines.append("| Preset | CPU % | RAM % | FPS | Detection Latency | Frame Latency |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        base_item = report["results"][base]
        b_cpu = float(base_item.get("cpu_percent", {}).get("avg", 0.0))
        b_mem = float(base_item.get("memory_percent", {}).get("avg", 0.0))
        b_fps = float(base_item.get("fps", {}).get("avg", 0.0))
        b_inf = float(base_item.get("inference_latency_ms", {}).get("avg", 0.0))
        b_frm = float(base_item.get("frame_latency_ms", {}).get("avg", 0.0))
        for preset in preset_order:
            item = report["results"].get(preset, {})
            cpu = float(item.get("cpu_percent", {}).get("avg", 0.0))
            mem = float(item.get("memory_percent", {}).get("avg", 0.0))
            fps = float(item.get("fps", {}).get("avg", 0.0))
            inf = float(item.get("inference_latency_ms", {}).get("avg", 0.0))
            frm = float(item.get("frame_latency_ms", {}).get("avg", 0.0))
            lines.append(
                f"| {preset} | "
                f"{_delta_pct(b_cpu, cpu):+.2f}% | "
                f"{_delta_pct(b_mem, mem):+.2f}% | "
                f"{_delta_pct(b_fps, fps):+.2f}% | "
                f"{_delta_pct(b_inf, inf):+.2f}% | "
                f"{_delta_pct(b_frm, frm):+.2f}% |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark runtime performance presets.")
    parser.add_argument("--presets", default="balanced,max_accuracy", help="Comma-separated preset names.")
    parser.add_argument("--sample-sec", type=float, default=40.0, help="Sampling duration per preset (seconds).")
    parser.add_argument("--warmup-sec", type=float, default=20.0, help="Warmup duration per preset (seconds).")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="Polling interval in seconds.")
    parser.add_argument("--manage-runtime", action="store_true", help="Start/stop runtime process automatically.")
    parser.add_argument("--runtime-python", default="", help="Optional python executable for runtime process.")
    parser.add_argument("--tag", default="", help="Optional suffix in report filename.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    state_file = root / "runtime" / "dashboard_state.json"
    override_file = root / "runtime" / "runtime_settings.override.json"
    reports_dir = root / "reports"
    logs_dir = root / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    requested = [p.strip().lower() for p in str(args.presets).split(",") if p.strip()]
    presets = [p for p in requested if p in PRESET_PATCHES]
    if not presets:
        print("No valid presets selected. Available: eco, balanced, max_accuracy")
        return 2

    backup_exists = override_file.exists()
    backup_text = ""
    if backup_exists:
        backup_text = override_file.read_text(encoding="utf-8")

    runtime_proc: Optional[subprocess.Popen] = None
    results: Dict[str, Any] = {}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_suffix = f"_{args.tag.strip()}" if args.tag.strip() else ""
    runtime_log = logs_dir / f"benchmark_runtime_{run_id}{tag_suffix}.log"
    runtime_python = _resolve_runtime_python(root, args.runtime_python)

    try:
        if args.manage_runtime:
            runtime_proc = _start_runtime(root, runtime_log, runtime_python)
            # Initial startup grace
            time.sleep(8.0)
            if runtime_proc.poll() is not None:
                raise RuntimeError(
                    f"Runtime exited early. Check log: {runtime_log}"
                )
            _wait_for_fresh_state(state_file, timeout_sec=25.0)

        for preset in presets:
            patch = PRESET_PATCHES[preset]
            _apply_override_patch(override_file, patch)
            print(f"[benchmark] preset={preset} patch applied")
            _wait_for_fresh_state(state_file, timeout_sec=max(8.0, float(args.warmup_sec)))
            time.sleep(max(0.5, float(args.warmup_sec)))
            samples = _collect_samples(
                state_file=state_file,
                duration_sec=float(args.sample_sec),
                poll_sec=float(args.poll_sec),
            )
            results[preset] = _summarize_samples(samples)
            print(f"[benchmark] preset={preset} samples={len(samples)}")

    finally:
        _stop_runtime(runtime_proc)
        if backup_exists:
            override_file.write_text(backup_text, encoding="utf-8")
        elif override_file.exists():
            try:
                override_file.unlink()
            except Exception:
                pass

    report = {
        "generated_at": datetime.now().isoformat(),
        "runtime_python": str(runtime_python),
        "sample_sec": float(args.sample_sec),
        "warmup_sec": float(args.warmup_sec),
        "poll_sec": float(args.poll_sec),
        "presets": presets,
        "results": results,
    }

    json_path = reports_dir / f"performance_benchmark_{run_id}{tag_suffix}.json"
    md_path = reports_dir / f"performance_benchmark_{run_id}{tag_suffix}.md"
    _write_json(json_path, report)
    md_path.write_text(_render_markdown(report, presets), encoding="utf-8")
    print(f"[benchmark] report_json={json_path}")
    print(f"[benchmark] report_md={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
