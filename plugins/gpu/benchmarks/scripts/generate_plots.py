#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Resolve directories relative to this script location (no hardcoded global paths)
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
data_dir = base_dir / "data"
plots_dir = base_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Load data
with open(data_dir / "exec_bench_stats.json") as f:
    nv_exec = json.load(f)
with open(data_dir / "compile_bench_stats.json") as f:
    nv_comp = json.load(f)
with open(data_dir / "cpu_amd_exec_stats.json") as f:
    cpu_amd_exec = json.load(f)
with open(data_dir / "cpu_amd_compile_stats.json") as f:
    cpu_amd_comp = json.load(f)

benchmarks = ["VecAdd", "SAXPY", "Classify"]
scales = ["1M", "10M", "100M"]
x_indices = np.arange(len(scales))

# ==============================================================================
# PLOT 1: NVIDIA L40S GPU (3 Toolchains: NVCC, PCUDA, Native SYCL)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14), gridspec_kw={'height_ratios': [1.1, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig.patch.set_facecolor('#ffffff')

nv_toolchains = ["Native NVCC", "ACPP PCUDA", "ACPP Native"]
width_nv = 0.25

# Row 1: Execution Latency Breakdown
for idx, b in enumerate(benchmarks):
    ax = axes[0, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    for t_idx, tc in enumerate(nv_toolchains):
        pure_gpu = []
        launch_ovh = []
        total_means = []
        total_stds = []
        for s in scales:
            d = nv_exec[b][s][tc]
            pure_gpu.append(d["pure_gpu_mean_us"])
            launch_ovh.append(d["overhead_mean_us"])
            total_means.append(d["total_mean_us"])
            total_stds.append(d["total_std_us"])

        pos = x_indices + (t_idx - 1) * width_nv
        b1 = ax.bar(pos, pure_gpu, width_nv, color='#3b82f6', edgecolor='#1d4ed8', linewidth=0.8,
                    label="Pure GPU Kernel Execution" if (idx == 0 and t_idx == 0) else "", zorder=3)
        b2 = ax.bar(pos, launch_ovh, width_nv, bottom=pure_gpu, yerr=total_stds, capsize=3.5,
                    color='#f97316', edgecolor='#c2410c', linewidth=0.8,
                    label="cudaLaunchKernel / Launch Overhead" if (idx == 0 and t_idx == 0) else "", zorder=4)

        for bar, tm, ts in zip(b2, total_means, total_stds):
            h = bar.get_height() + bar.get_y()
            txt = f"{tm:.1f}µs\n±{ts:.1f}µs" if tm < 1000 else f"{tm/1000.0:.2f}ms\n±{ts:.0f}µs"
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 4.0)
    ax.set_title(f"{b} — Execution Latency Breakdown", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{s}\nElements" for s in scales], fontsize=11, fontweight='medium')
    ax.set_ylabel("Execution Latency (µs, Log Scale)", fontsize=11, fontweight='bold')
    if idx == 0:
        ax.legend(loc='upper left', framealpha=0.95, fontsize=10)

# Row 2: Compilation Time
comp_colors_nv = ["#20c997", "#e67700", "#845ef7"]
for idx, b in enumerate(benchmarks):
    ax = axes[1, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    means = [nv_comp[b][tc]["mean_ms"] for tc in nv_toolchains]
    stds = [nv_comp[b][tc]["std_ms"] for tc in nv_toolchains]
    x_pos = np.arange(len(nv_toolchains))

    bars = ax.bar(x_pos, means, width=0.55, yerr=stds, capsize=5, color=comp_colors_nv,
                  edgecolor='#222222', linewidth=0.8, zorder=3)

    for bar, m, s, tc in zip(bars, means, stds, nv_toolchains):
        h = bar.get_height()
        speedup = ""
        if tc == "ACPP PCUDA":
            speedup = f"\n[{means[0]/m:.1f}x faster vs NVCC]"
        label_text = f"{m/1000.0:.2f} s\n({m:.0f} ± {s:.0f} ms){speedup}"
        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax.set_title(f"{b} — JIT Compilation Time", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(nv_toolchains, fontsize=10.5, fontweight='bold')
    ax.set_ylabel("Compilation Time (ms, Linear Scale)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 8200)

fig.suptitle("NVIDIA L40S: Execution Latency Breakdown & JIT Compilation Time\nNative NVCC vs. AdaptiveCpp PCUDA vs. AdaptiveCpp Native (Mean ± 1σ)",
             fontsize=16, fontweight='heavy', y=0.98)

out_nv = plots_dir / "nvidia_l40s_benchmark.png"
plt.savefig(out_nv, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {out_nv}")


# ==============================================================================
# PLOT 2: AMD INSTINCT MI210 GPU (2 Toolchains: AMD PCUDA, AMD ACPP Native)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14), gridspec_kw={'height_ratios': [1.1, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig.patch.set_facecolor('#ffffff')

amd_toolchains = ["AMD PCUDA", "AMD ACPP Native"]
amd_colors = ["#e67700", "#7950f2"]
width_2 = 0.32

# Row 1: Execution Latency
for idx, b in enumerate(benchmarks):
    ax = axes[0, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    for t_idx, tc in enumerate(amd_toolchains):
        means = []
        stds = []
        for s in scales:
            d = cpu_amd_exec["AMD_GPU"][tc][b][s]
            means.append(d["mean_us"])
            stds.append(d["std_us"])

        pos = x_indices + (t_idx - 0.5) * width_2
        bars = ax.bar(pos, means, width_2, yerr=stds, capsize=4, label=tc,
                      color=amd_colors[t_idx], edgecolor='#222222', linewidth=0.8, zorder=3)

        for bar, m, s in zip(bars, means, stds):
            h = bar.get_height()
            txt = f"{m:.1f}µs\n±{s:.1f}" if m < 1000 else f"{m/1000.0:.2f}ms\n±{s:.0f}µs"
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 4.0)
    ax.set_title(f"{b} — Execution Latency", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{s}\nElements" for s in scales], fontsize=11, fontweight='medium')
    ax.set_ylabel("Execution Latency (µs, Log Scale)", fontsize=11, fontweight='bold')
    if idx == 0:
        ax.legend(loc='upper left', framealpha=0.95, fontsize=11)

# Row 2: Compilation Time
amd_comp_tc_map = {
    "AMD PCUDA": "ACPP PCUDA",
    "AMD ACPP Native": "ACPP Native (SYCL)"
}
for idx, b in enumerate(benchmarks):
    ax = axes[1, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    means = [cpu_amd_comp[amd_comp_tc_map[tc]][b]["mean"] for tc in amd_toolchains]
    stds = [cpu_amd_comp[amd_comp_tc_map[tc]][b]["std"] for tc in amd_toolchains]
    x_pos = np.arange(len(amd_toolchains))

    bars = ax.bar(x_pos, means, width=0.48, yerr=stds, capsize=5, color=amd_colors,
                  edgecolor='#222222', linewidth=0.8, zorder=3)

    for bar, m, s, tc in zip(bars, means, stds, amd_toolchains):
        h = bar.get_height()
        speedup = ""
        if tc == "AMD PCUDA":
            speedup = f"\n[{means[1]/m:.1f}x faster vs SYCL]"
        label_text = f"{m/1000.0:.2f} s\n({m:.0f} ± {s:.0f} ms){speedup}"
        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f"{b} — JIT Compilation Time", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(amd_toolchains, fontsize=11, fontweight='bold')
    ax.set_ylabel("Compilation Time (ms, Linear Scale)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 7800)

fig.suptitle("AMD Instinct MI210: Execution Latency & JIT Compilation Time\nNautilus AdaptiveCpp PCUDA vs. Nautilus AdaptiveCpp Native SYCL (Mean ± 1σ)",
             fontsize=16, fontweight='heavy', y=0.98)

out_amd = plots_dir / "amd_mi210_benchmark.png"
plt.savefig(out_amd, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {out_amd}")


# ==============================================================================
# PLOT 3: INTEL XEON PLATINUM 8454H CPU (2 Toolchains: PCUDA CPU, Native SYCL CPU)
# ==============================================================================
fig, axes = plt.subplots(2, 3, figsize=(22, 14), gridspec_kw={'height_ratios': [1.1, 1.0], 'hspace': 0.38, 'wspace': 0.22})
fig.patch.set_facecolor('#ffffff')

cpu_toolchains = ["ACPP PCUDA (CPU)", "ACPP Native (CPU)"]
cpu_colors = ["#e67700", "#7950f2"]

# Row 1: Execution Latency
for idx, b in enumerate(benchmarks):
    ax = axes[0, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    for t_idx, tc in enumerate(cpu_toolchains):
        means = []
        stds = []
        for s in scales:
            d = cpu_amd_exec["CPU"][tc][b][s]
            means.append(d["mean_us"])
            stds.append(d["std_us"])

        pos = x_indices + (t_idx - 0.5) * width_2
        bars = ax.bar(pos, means, width_2, yerr=stds, capsize=4, label=tc,
                      color=cpu_colors[t_idx], edgecolor='#222222', linewidth=0.8, zorder=3)

        for bar, m, s in zip(bars, means, stds):
            h = bar.get_height()
            txt = f"{m:.1f}µs\n±{s:.1f}" if m < 1000 else f"{m/1000.0:.2f}ms\n±{s/1000.0:.2f}ms"
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 6), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_yscale('log')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 4.0)
    ax.set_title(f"{b} — Execution Latency", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{s}\nElements" for s in scales], fontsize=11, fontweight='medium')
    ax.set_ylabel("Execution Latency (µs, Log Scale)", fontsize=11, fontweight='bold')
    if idx == 0:
        ax.legend(loc='upper left', framealpha=0.95, fontsize=11)

# Row 2: Compilation Time
cpu_comp_tc_map = {
    "ACPP PCUDA (CPU)": "ACPP PCUDA",
    "ACPP Native (CPU)": "ACPP Native (SYCL)"
}
for idx, b in enumerate(benchmarks):
    ax = axes[1, idx]
    ax.set_facecolor('#fdfdfd')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e5e5e5', zorder=0)

    means = [cpu_amd_comp[cpu_comp_tc_map[tc]][b]["mean"] for tc in cpu_toolchains]
    stds = [cpu_amd_comp[cpu_comp_tc_map[tc]][b]["std"] for tc in cpu_toolchains]
    x_pos = np.arange(len(cpu_toolchains))

    bars = ax.bar(x_pos, means, width=0.48, yerr=stds, capsize=5, color=cpu_colors,
                  edgecolor='#222222', linewidth=0.8, zorder=3)

    for bar, m, s, tc in zip(bars, means, stds, cpu_toolchains):
        h = bar.get_height()
        speedup = ""
        if tc == "ACPP PCUDA (CPU)":
            speedup = f"\n[{means[1]/m:.1f}x faster vs SYCL]"
        label_text = f"{m/1000.0:.2f} s\n({m:.0f} ± {s:.0f} ms){speedup}"
        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f"{b} — JIT Compilation Time", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cpu_toolchains, fontsize=11, fontweight='bold')
    ax.set_ylabel("Compilation Time (ms, Linear Scale)", fontsize=11, fontweight='bold')
    ax.set_ylim(0, 7800)

fig.suptitle("Intel Xeon Platinum 8454H (Host CPU): Execution Latency & JIT Compilation Time\nNautilus AdaptiveCpp PCUDA vs. Nautilus AdaptiveCpp Native SYCL (Mean ± 1σ)",
             fontsize=16, fontweight='heavy', y=0.98)

out_cpu = plots_dir / "intel_xeon_cpu_benchmark.png"
plt.savefig(out_cpu, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved {out_cpu}")
