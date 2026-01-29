#!/usr/bin/env python3
"""
ISO-Bench Pipeline Diagram - Figure 1
Publication-quality pipeline diagram for academic paper.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import matplotlib.patheffects as pe

# Configure for academic publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'text.usetex': False,
})

# Set up figure (single column width ~3.5in, double ~7in)
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 11)
ax.axis('off')

# Academic color scheme (muted, professional)
COLORS = {
    'source': '#D4E6F1',       # Soft blue
    'process': '#FCF3CF',      # Soft yellow
    'output': '#D5F5E3',       # Soft green
    'agent': '#FADBD8',        # Soft red/pink
    'metric': '#E8DAEF',       # Soft purple
    'border': '#2C3E50',       # Dark blue-gray
    'arrow': '#566573',        # Medium gray
    'text': '#1B2631',         # Near black
    'subtext': '#5D6D7E',      # Gray
}


def draw_box(ax, x, y, width, height, text, color, fontsize=8.5,
             subtext=None, bold=False):
    """Draw a professional rounded box."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.08",
        facecolor=color,
        edgecolor=COLORS['border'],
        linewidth=1.0
    )
    ax.add_patch(box)

    weight = 'bold' if bold else 'medium'
    if subtext:
        ax.text(x, y + 0.12, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=COLORS['text'])
        ax.text(x, y - 0.18, subtext, ha='center', va='center',
                fontsize=7, color=COLORS['subtext'], style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=COLORS['text'])


def draw_arrow(ax, start, end, label=None, style='-', shrink=8):
    """Draw a professional arrow."""
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle='-|>',
            color=COLORS['arrow'],
            lw=1.0,
            linestyle=style,
            shrinkA=shrink,
            shrinkB=shrink,
            mutation_scale=10
        )
    )
    if label:
        mid_x = (start[0] + end[0]) / 2 + 0.25
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, label, fontsize=7, color=COLORS['subtext'],
                ha='left', va='center')


# Section headers with underlines
ax.text(3.5, 10.3, 'Benchmark Construction', fontsize=11, fontweight='bold',
        ha='center', color=COLORS['text'])
ax.plot([1.5, 5.5], [10.05, 10.05], color=COLORS['border'], lw=0.8)

ax.text(11, 10.3, 'Evaluation Framework', fontsize=11, fontweight='bold',
        ha='center', color=COLORS['text'])
ax.plot([8.8, 13.2], [10.05, 10.05], color=COLORS['border'], lw=0.8)


# ============== LEFT: Benchmark Construction ==============

# Step 1: Source
draw_box(ax, 3.5, 9.2, 3.5, 0.65, 'Source Repositories', COLORS['source'],
         subtext='vLLM, SGLang')

# Step 2: Commit Extraction
draw_arrow(ax, (3.5, 8.87), (3.5, 8.33))
draw_box(ax, 3.5, 8.0, 3.5, 0.65, 'Commit Extraction', COLORS['process'],
         subtext='Performance-related keywords')

# Step 3: Manual Curation
draw_arrow(ax, (3.5, 7.67), (3.5, 7.13))
draw_box(ax, 3.5, 6.8, 3.5, 0.65, 'Manual Curation', COLORS['process'],
         subtext='Hardware compat., benchmark validity')

# Step 4: Test Generation
draw_arrow(ax, (3.5, 6.47), (3.5, 5.93))
draw_box(ax, 3.5, 5.6, 3.5, 0.65, 'LLM Test Generation', COLORS['process'],
         subtext='GPT-4 with syntax repair')

# Step 5: Validation
draw_arrow(ax, (3.5, 5.27), (3.5, 4.73))
draw_box(ax, 3.5, 4.4, 3.5, 0.65, 'Execution Validation', COLORS['process'],
         subtext='Base, optimized, main branch')

# Output
draw_arrow(ax, (3.5, 4.07), (3.5, 3.53))
draw_box(ax, 3.5, 3.2, 2.8, 0.65, 'Benchmark Tasks', COLORS['output'],
         bold=True, subtext='39 vLLM + 15 SGLang')


# ============== RIGHT: Evaluation Framework ==============

# Step 1: Task Spec
draw_box(ax, 11, 9.2, 3.8, 0.65, 'Task Specification', COLORS['source'],
         subtext='commit, prompt, perf. command')

# Connect left to right (dashed)
ax.annotate(
    '', xy=(9.0, 9.2), xytext=(5.3, 3.2),
    arrowprops=dict(
        arrowstyle='-|>',
        color='#ABB2B9',
        lw=0.8,
        linestyle='--',
        connectionstyle='arc3,rad=0.15',
        shrinkA=12,
        shrinkB=12,
        mutation_scale=8
    )
)

# Step 2: Agent
draw_arrow(ax, (11, 8.87), (11, 8.33))
draw_box(ax, 11, 8.0, 3.8, 0.65, 'Agent Execution', COLORS['agent'],
         subtext='Codex, TRAE, OpenHands, Claude Code')

# Step 3: Patch
draw_arrow(ax, (11, 7.67), (11, 7.13))
draw_box(ax, 11, 6.8, 3.8, 0.65, 'Patch Generation', COLORS['agent'],
         subtext='Code modifications')

# Step 4: Execution
draw_arrow(ax, (11, 6.47), (11, 5.93))
draw_box(ax, 11, 5.6, 3.8, 0.65, 'Isolated Execution', COLORS['process'],
         subtext='Docker / Modal (H100)')

# Benchmark types - branch out
draw_arrow(ax, (9.8, 5.27), (9.3, 4.73))
draw_arrow(ax, (11, 5.27), (11, 4.73))
draw_arrow(ax, (12.2, 5.27), (12.7, 4.73))

draw_box(ax, 9.3, 4.4, 1.7, 0.55, 'Serving', COLORS['metric'],
         fontsize=8, subtext='TTFT')
draw_box(ax, 11, 4.4, 1.7, 0.55, 'Throughput', COLORS['metric'],
         fontsize=8, subtext='tok/s')
draw_box(ax, 12.7, 4.4, 1.7, 0.55, 'Latency', COLORS['metric'],
         fontsize=8, subtext='ms')

# Converge
draw_arrow(ax, (9.3, 4.12), (10.2, 3.58))
draw_arrow(ax, (11, 4.12), (11, 3.58))
draw_arrow(ax, (12.7, 4.12), (11.8, 3.58))

# 3-way comparison
draw_box(ax, 11, 3.2, 3.8, 0.65, '3-Way Comparison', COLORS['process'],
         subtext='Baseline vs. Human vs. Agent')

# Final metrics
draw_arrow(ax, (11, 2.87), (11, 2.33))
draw_box(ax, 11, 2.0, 3.2, 0.55, 'Performance Metrics', COLORS['output'],
         bold=True, subtext='Opt@K, \u0394TTFT, \u0394Throughput')


# Filtering numbers on left side (small annotations)
ax.text(5.4, 8.6, '23K+', fontsize=6.5, color=COLORS['subtext'], ha='left')
ax.text(5.4, 7.4, '~500', fontsize=6.5, color=COLORS['subtext'], ha='left')
ax.text(5.4, 6.2, '~100', fontsize=6.5, color=COLORS['subtext'], ha='left')
ax.text(5.4, 5.0, '54', fontsize=6.5, color=COLORS['subtext'], ha='left')


# Minimal legend (bottom right)
legend_items = [
    ('Input/Output', COLORS['source'], COLORS['output']),
    ('Processing', COLORS['process']),
    ('Agent', COLORS['agent']),
    ('Metrics', COLORS['metric']),
]

legend_x = 0.8
legend_y = 1.8
for i, item in enumerate(legend_items):
    y_pos = legend_y - i * 0.35
    if len(item) == 3:
        # Two-color item
        box1 = FancyBboxPatch((legend_x, y_pos - 0.1), 0.2, 0.2,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              facecolor=item[1], edgecolor=COLORS['border'], lw=0.5)
        box2 = FancyBboxPatch((legend_x + 0.22, y_pos - 0.1), 0.2, 0.2,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              facecolor=item[2], edgecolor=COLORS['border'], lw=0.5)
        ax.add_patch(box1)
        ax.add_patch(box2)
        ax.text(legend_x + 0.55, y_pos, item[0], fontsize=7, va='center',
                color=COLORS['text'])
    else:
        box = FancyBboxPatch((legend_x, y_pos - 0.1), 0.2, 0.2,
                             boxstyle="round,pad=0.01,rounding_size=0.03",
                             facecolor=item[1], edgecolor=COLORS['border'], lw=0.5)
        ax.add_patch(box)
        ax.text(legend_x + 0.35, y_pos, item[0], fontsize=7, va='center',
                color=COLORS['text'])


plt.tight_layout(pad=0.5)
plt.savefig('scripts/figures/pipeline_diagram.pdf',
            dpi=300, bbox_inches='tight', facecolor='white',
            backend='pdf')
plt.savefig('scripts/figures/pipeline_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print("Saved: scripts/figures/pipeline_diagram.pdf and .png")
