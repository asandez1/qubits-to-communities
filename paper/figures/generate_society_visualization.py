#!/usr/bin/env python3
"""Generate the society-model visualization.

The default run produces only the publication figure (Option B,
multi-panel) at figures/fig4_society.png. Two alternate visualizations
are kept as functions for talks/slides but not invoked by default:
Option A (single rich 3D) and Option C (hybrid 3D + insets).

Run from paper directory:
    python3 figures/generate_society_visualization.py            # → fig4_society.png
    python3 figures/generate_society_visualization.py --all      # also writes A and C
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 130

# =====================================================================
# Synthetic snapshot data — represents M-tier across 4 cycles
# =====================================================================
#
# 12 members (matches benchmark_fixtures.tier_m). For visualization, we
# generate plausible balance/energy trajectories so the "evolution" story
# is visible across cycles.

MEMBERS = [
    # id,    age, x, y,  skill,        vehicle, household
    ("E1",   62, 1, 9,  "electrical", "van",   "hh_e1"),
    ("E2",   26, 8, 8,  "electrical", "bike",  "hh_e1"),
    ("E3",   42, 5, 5,  "electrical", "van",   "hh_e3"),
    ("Tu1",  68, 2, 2,  "tutoring",   "car",   "hh_tu1"),
    ("Tu2",  28, 9, 3,  "tutoring",   "bike",  "hh_fam1"),
    ("Tu3",  17, 5, 1,  "tutoring",   "none",  "hh_teen"),
    ("Tu4",  38, 7, 7,  "tutoring",   "car",   "hh_fam2"),
    ("Co1",  58, 3, 1,  "cooking",    "car",   "hh_fam1"),
    ("Co2",  44, 8, 2,  "cooking",    "bike",  "hh_co2"),
    ("Co3",  23, 5, 9,  "cooking",    "bike",  "hh_e3"),
    ("G1",   35, 4, 5,  "generalist", "car",   "hh_g1"),
    ("G2",   48, 6, 6,  "generalist", "car",   "hh_fam2"),
]

SKILL_COLORS = {
    "electrical": "#2874A6",
    "tutoring":   "#27AE60",
    "cooking":    "#E67E22",
    "generalist": "#7F8C8D",
}
HOUSEHOLD_COLORS = {
    "hh_e1":    "#5DADE2",
    "hh_e3":    "#A569BD",
    "hh_tu1":   "#58D68D",
    "hh_fam1":  "#F1948A",
    "hh_fam2":  "#F8C471",
    "hh_teen":  "#85C1E9",
    "hh_co2":   "#F0B27A",
    "hh_g1":    "#BDC3C7",
}
VEHICLE_MARKER = {"none": "o", "bike": "s", "car": "^", "van": "D"}


def age_to_capacity(age):
    if age < 12: return 2.0
    elif age <= 18: return 4.0 + (age - 12) * (3.0 / 6.0)
    elif age <= 30: return 10.0
    elif age <= 50: return 10.0 - (age - 30) * (3.0 / 20.0)
    elif age <= 65: return 7.0 - (age - 50) * (3.0 / 15.0)
    elif age <= 80: return 4.0 - (age - 65) * (2.0 / 15.0)
    else: return 1.0


def synth_snapshots(n_cycles=4):
    """Generate 4 cycles of plausible state evolution."""
    rng = np.random.default_rng(42)
    snapshots = []
    balances = {m[0]: 50.0 for m in MEMBERS}
    energies = {m[0]: age_to_capacity(m[1]) for m in MEMBERS}

    for c in range(n_cycles):
        # Members with higher skill in scarce categories (electrical) earn more
        snap = []
        for mid, age, x, y, skill, vehicle, hh in MEMBERS:
            cap = age_to_capacity(age)
            # Earn rate by category scarcity
            earn = {"electrical": rng.normal(20, 5),
                    "cooking": rng.normal(10, 3),
                    "tutoring": rng.normal(8, 3),
                    "generalist": rng.normal(12, 4)}.get(skill, 5)
            balances[mid] = max(0, balances[mid] + earn - rng.normal(6, 2))
            # Energy depletes during the day, recovers overnight (age-modulated)
            depletion = rng.uniform(0.4, 0.9) * cap
            recovery_rate = max(0.4, 1.0 - max(0, age - 30) * 0.012)
            energies[mid] = min(cap, max(0, cap - depletion) + recovery_rate * depletion)
            snap.append(dict(
                member_id=mid, age=age, x=x, y=y, skill=skill,
                vehicle=vehicle, household=hh,
                capacity=cap, energy=energies[mid],
                balance=balances[mid],
            ))
        snapshots.append(snap)
    return snapshots


SNAPSHOTS = synth_snapshots(n_cycles=4)
CYCLES = [0, 10, 20, 30]


def member_glyph_size(age, scale=1.0):
    return 30 + age_to_capacity(age) * 18 * scale


# =====================================================================
# OPTION A: Single rich 3D figure
# =====================================================================

def option_a_rich_3d():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Z positions for the 4 snapshot planes
    z_planes = [0, 1, 2, 3]

    # Draw translucent planes
    for z in z_planes:
        xx, yy = np.meshgrid([0, 10], [0, 10])
        zz = np.full_like(xx, z, dtype=float)
        ax.plot_surface(xx, yy, zz, alpha=0.06, color="lightblue", edgecolor="none")

    # Plot members on each plane
    for plane_idx, (z, snap) in enumerate(zip(z_planes, SNAPSHOTS)):
        # Household halos (drawn behind members)
        hh_groups = {}
        for m in snap:
            hh_groups.setdefault(m["household"], []).append(m)
        for hid, mlist in hh_groups.items():
            if len(mlist) <= 1: continue
            # Draw connecting lines for households (3D)
            xs = [m["x"] for m in mlist] + [mlist[0]["x"]]
            ys = [m["y"] for m in mlist] + [mlist[0]["y"]]
            zs = [z] * (len(mlist) + 1)
            ax.plot(xs, ys, zs, color=HOUSEHOLD_COLORS[hid], lw=1.5, alpha=0.5)

        # Members
        for m in snap:
            color = SKILL_COLORS[m["skill"]]
            alpha = 0.4 + 0.6 * (m["energy"] / max(m["capacity"], 0.1))
            marker = VEHICLE_MARKER[m["vehicle"]]
            ax.scatter([m["x"]], [m["y"]], [z],
                       s=member_glyph_size(m["age"]),
                       c=color, marker=marker, alpha=alpha,
                       edgecolors="black", linewidths=0.8, depthshade=True)

    # Trajectories (vertical lines for each member across cycles)
    for mid, age, x, y, skill, vehicle, hh in MEMBERS:
        zs = z_planes
        balances = [s[i]["balance"] for i, s in enumerate(
            [SNAPSHOTS[k] for k in range(4)]
        ) for sn in [s] for ss in [m for m in s if m["member_id"] == mid]] if False else \
            [next(m["balance"] for m in s if m["member_id"] == mid) for s in SNAPSHOTS]
        # Color trajectory by balance trajectory (green = growing)
        delta = balances[-1] - balances[0]
        traj_color = "#27AE60" if delta > 30 else ("#7F8C8D" if abs(delta) < 30 else "#E74C3C")
        ax.plot([x] * 4, [y] * 4, zs, color=traj_color, lw=1, alpha=0.4)

    # Snapshot file icons + cycle labels
    for z, c in zip(z_planes, CYCLES):
        ax.text(11.5, 0, z, f"snap_{c:02d}.json",
                color="#1A5276", fontsize=9, fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB",
                          edgecolor="#5DADE2"))
        ax.text(-1.5, 5, z, f"cycle {c}", color="dimgray", fontsize=10,
                ha="right", va="center", fontweight="bold")

    # Cycle engine arrows between planes (vertical "→")
    for i in range(len(z_planes) - 1):
        ax.text(5, -1.5, (z_planes[i] + z_planes[i+1]) / 2,
                "↑ CycleEngine.advance()", color="#7D3C98", fontsize=8,
                ha="center", va="center", style="italic")

    ax.set_xlabel("Location X (km)", fontsize=10)
    ax.set_ylabel("Location Y (km)", fontsize=10)
    ax.set_zlabel("Time →", fontsize=10)
    ax.set_xlim(0, 13)
    ax.set_ylim(-1, 11)
    ax.set_zticks(z_planes)
    ax.set_zticklabels([f"c={c}" for c in CYCLES])
    ax.view_init(elev=22, azim=-50)

    # Title
    fig.suptitle("OrquestIA Society Model — One Glance",
                 fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.5, 0.91,
             "Members glyph: size=age-energy capacity, color=skill, shape=vehicle, alpha=remaining energy. "
             "Households connected by colored lines. Vertical trajectories = same member over cycles.",
             ha="center", fontsize=9, style="italic", color="dimgray")

    # Legend (small, top-right)
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SKILL_COLORS["electrical"],
               markersize=10, label="Electrical"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SKILL_COLORS["tutoring"],
               markersize=10, label="Tutoring"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SKILL_COLORS["cooking"],
               markersize=10, label="Cooking"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SKILL_COLORS["generalist"],
               markersize=10, label="Generalist"),
        Line2D([0], [0], marker="o", color="gray", markersize=10, label="Vehicle: none"),
        Line2D([0], [0], marker="s", color="gray", markersize=10, label="Vehicle: bike"),
        Line2D([0], [0], marker="^", color="gray", markersize=10, label="Vehicle: car"),
        Line2D([0], [0], marker="D", color="gray", markersize=10, label="Vehicle: van"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.02, 0.98),
              fontsize=8, ncol=2, framealpha=0.9, title="Glyph Encoding")

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "society_optionA_3d_rich.png"), bbox_inches="tight")
    plt.close(fig)
    print("  wrote society_optionA_3d_rich.png")


# =====================================================================
# OPTION B: Multi-panel composition
# =====================================================================

def option_b_multipanel():
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)

    # --- Panel 1: Glyph anatomy (left, spans 1 row) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 4)
    ax1.axis("off")
    ax1.set_title("Member Glyph", fontsize=11, fontweight="bold")
    # Big sample member
    ax1.scatter([2], [2.2], s=350, c=SKILL_COLORS["electrical"],
                marker="D", edgecolors="black", linewidths=1.5, alpha=0.85)
    # Halo for household
    halo = Circle((2, 2.2), 0.6, facecolor="none",
                  edgecolor=HOUSEHOLD_COLORS["hh_e1"], linewidth=2.5, linestyle="--")
    ax1.add_patch(halo)
    # Annotations
    ax1.annotate("size = age-derived\nenergy capacity",
                 xy=(2.2, 2.5), xytext=(2.7, 3.6),
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax1.annotate("color = primary skill\n(electrical)",
                 xy=(1.8, 2.2), xytext=(0.05, 3.3),
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax1.annotate("shape = vehicle\n(◇ van)",
                 xy=(2, 1.9), xytext=(0.05, 0.3),
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax1.annotate("dashed circle\n= household",
                 xy=(2.55, 2.55), xytext=(2.7, 0.6),
                 fontsize=8, ha="left",
                 arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    # --- Panel 2: Top-down map of one snapshot (cycle 0, spans 2 rows, 2 cols) ---
    ax2 = fig.add_subplot(gs[0:2, 1:3])
    ax2.set_title("Snapshot at Cycle 0 — Geographic View",
                  fontsize=11, fontweight="bold")
    snap0 = SNAPSHOTS[0]
    # Households
    hh_groups = {}
    for m in snap0:
        hh_groups.setdefault(m["household"], []).append(m)
    for hid, mlist in hh_groups.items():
        if len(mlist) <= 1: continue
        xs = [m["x"] for m in mlist]
        ys = [m["y"] for m in mlist]
        # Centroid + circle
        cx, cy = np.mean(xs), np.mean(ys)
        radius = max(0.6, max(np.hypot(np.array(xs) - cx, np.array(ys) - cy)) + 0.4)
        ax2.add_patch(Circle((cx, cy), radius, facecolor=HOUSEHOLD_COLORS[hid],
                              alpha=0.18, edgecolor=HOUSEHOLD_COLORS[hid],
                              linestyle="--", linewidth=1.5))
    # Members
    for m in snap0:
        ax2.scatter([m["x"]], [m["y"]],
                    s=member_glyph_size(m["age"]) * 1.5,
                    c=SKILL_COLORS[m["skill"]],
                    marker=VEHICLE_MARKER[m["vehicle"]],
                    edgecolors="black", linewidths=1.0, alpha=0.85)
        ax2.text(m["x"] + 0.25, m["y"] + 0.25, m["member_id"],
                 fontsize=8, fontweight="bold")
    # Sample tasks
    tasks = [
        (2, 9, "elec_1\n(urgent)", "#2874A6"),
        (7, 3, "elec_2", "#2874A6"),
        (4, 9, "cook_1\n(urgent)", "#E67E22"),
        (3, 3, "tutor_2", "#27AE60"),
    ]
    for tx, ty, label, color in tasks:
        ax2.plot(tx, ty, marker="X", color=color, markersize=15,
                 markeredgecolor="black", markeredgewidth=1.5)
        ax2.text(tx, ty - 0.7, label, fontsize=7, ha="center",
                 color=color, fontweight="bold")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_xlabel("Location X (km)", fontsize=9)
    ax2.set_ylabel("Location Y (km)", fontsize=9)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.3)
    ax2.text(0.5, -1.2, "● Members  |  ✕ Tasks  |  Dashed = household",
             fontsize=8, color="dimgray", style="italic", transform=ax2.transData)

    # --- Panel 3: Cycle engine pipeline (top-right) ---
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis("off")
    ax3.set_title("CycleEngine.advance()", fontsize=11, fontweight="bold")
    steps = [
        (0.5, 8, "Snapshot N\n(frozen)", "#D4E6F1"),
        (0.5, 6.5, "1. Credit transfers", "#FAD7A0"),
        (0.5, 5.3, "1b. Household pool", "#FAD7A0"),
        (0.5, 4.1, "2. Reputation +", "#FAD7A0"),
        (0.5, 2.9, "3. Demurrage", "#FAD7A0"),
        (0.5, 1.7, "4. Energy recovery\n   (age-based)", "#FAD7A0"),
        (0.5, 0.3, "Snapshot N+1\n→ snapshots.jsonl", "#ABEBC6"),
    ]
    for x, y, label, color in steps:
        ax3.add_patch(FancyBboxPatch((x, y - 0.2), 9, 0.9,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, edgecolor="black", lw=0.8))
        ax3.text(x + 4.5, y + 0.25, label, ha="center", va="center",
                 fontsize=7.5, fontweight="bold")

    # --- Panel 4: Stacked snapshot evolution (bottom-left) ---
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.set_title("Snapshot Stack (4 cycles)",
                  fontsize=11, fontweight="bold")
    # Show 4 mini-snapshots side by side as cards
    for i, (snap, c) in enumerate(zip(SNAPSHOTS, CYCLES)):
        x_off = i * 2.5
        # Mini map background
        ax4.add_patch(Rectangle((x_off, 0), 2.0, 2.0,
                                 facecolor="#EBF5FB", edgecolor="#5DADE2", lw=1))
        # Members as small dots
        for m in snap:
            mx = x_off + 0.1 + (m["x"] / 10) * 1.8
            my = 0.1 + (m["y"] / 10) * 1.8
            energy_alpha = 0.4 + 0.6 * (m["energy"] / max(m["capacity"], 0.1))
            ax4.plot(mx, my, marker=VEHICLE_MARKER[m["vehicle"]],
                     color=SKILL_COLORS[m["skill"]], markersize=4,
                     alpha=energy_alpha, markeredgecolor="black",
                     markeredgewidth=0.3)
        ax4.text(x_off + 1.0, 2.3, f"cycle {c}", ha="center",
                 fontsize=9, fontweight="bold", color="#1A5276")
        # Arrow between snapshots
        if i < 3:
            ax4.annotate("", xy=(x_off + 2.4, 1.0), xytext=(x_off + 2.05, 1.0),
                         arrowprops=dict(arrowstyle="->", color="#7D3C98", lw=1.5))
    ax4.set_xlim(-0.3, 10.5)
    ax4.set_ylim(-0.5, 3)
    ax4.axis("off")

    # --- Panel 5: One-member trajectory (bottom-right) ---
    ax5 = fig.add_subplot(gs[2, 2:4])
    ax5.set_title("History Query: balance_series('E1')",
                  fontsize=11, fontweight="bold")
    # E1 balance trajectory (synthetic)
    e1_balances = [next(m["balance"] for m in s if m["member_id"] == "E1")
                   for s in SNAPSHOTS]
    e2_balances = [next(m["balance"] for m in s if m["member_id"] == "E2")
                   for s in SNAPSHOTS]
    tu1_balances = [next(m["balance"] for m in s if m["member_id"] == "Tu1")
                    for s in SNAPSHOTS]
    n1_balances = [next(m["balance"] for m in s if m["member_id"] == "Tu3")
                   for s in SNAPSHOTS]
    ax5.plot(CYCLES, e1_balances, marker="D", color=SKILL_COLORS["electrical"],
             label="E1 (age 62, electrical, van)", linewidth=2)
    ax5.plot(CYCLES, e2_balances, marker="s", color="#5DADE2",
             label="E2 (age 26, electrical, bike)", linewidth=2)
    ax5.plot(CYCLES, tu1_balances, marker="^", color=SKILL_COLORS["tutoring"],
             label="Tu1 (age 68, tutoring, car)", linewidth=2)
    ax5.plot(CYCLES, n1_balances, marker="o", color=SKILL_COLORS["generalist"],
             label="Tu3 (age 17, tutoring, none)", linewidth=2)
    ax5.set_xlabel("Cycle", fontsize=9)
    ax5.set_ylabel("Credit balance", fontsize=9)
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(alpha=0.3)

    fig.suptitle("The OrquestIA Society Model — Multi-Panel View",
                 fontsize=14, fontweight="bold", y=0.99)

    fig.savefig(os.path.join(HERE, "fig4_society.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig4_society.png")


# =====================================================================
# OPTION C: Hybrid — 3D centerpiece + 2 insets
# =====================================================================

def option_c_hybrid():
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.05, wspace=0.05,
                           width_ratios=[2.4, 2.4, 0.9, 0.9],
                           height_ratios=[1, 1, 1])

    # --- Main 3D view (left, large) ---
    ax_main = fig.add_subplot(gs[:, 0:2], projection="3d")

    z_planes = [0, 1, 2, 3]
    for z in z_planes:
        xx, yy = np.meshgrid([0, 10], [0, 10])
        zz = np.full_like(xx, z, dtype=float)
        ax_main.plot_surface(xx, yy, zz, alpha=0.05,
                              color="lightblue", edgecolor="none")

    for z, snap in zip(z_planes, SNAPSHOTS):
        # Household connector lines
        hh_groups = {}
        for m in snap:
            hh_groups.setdefault(m["household"], []).append(m)
        for hid, mlist in hh_groups.items():
            if len(mlist) <= 1: continue
            xs = [m["x"] for m in mlist] + [mlist[0]["x"]]
            ys = [m["y"] for m in mlist] + [mlist[0]["y"]]
            zs = [z] * (len(mlist) + 1)
            ax_main.plot(xs, ys, zs, color=HOUSEHOLD_COLORS[hid],
                          lw=1.5, alpha=0.55)

        for m in snap:
            color = SKILL_COLORS[m["skill"]]
            alpha = 0.4 + 0.6 * (m["energy"] / max(m["capacity"], 0.1))
            ax_main.scatter([m["x"]], [m["y"]], [z],
                             s=member_glyph_size(m["age"]) * 0.85,
                             c=color, marker=VEHICLE_MARKER[m["vehicle"]],
                             alpha=alpha, edgecolors="black", linewidths=0.7)

    # Trajectories
    for mid, age, x, y, skill, vehicle, hh in MEMBERS:
        balances = [next(m["balance"] for m in s if m["member_id"] == mid)
                    for s in SNAPSHOTS]
        delta = balances[-1] - balances[0]
        traj_color = "#27AE60" if delta > 30 else (
            "#7F8C8D" if abs(delta) < 30 else "#E74C3C")
        ax_main.plot([x] * 4, [y] * 4, z_planes,
                      color=traj_color, lw=0.8, alpha=0.45)

    # Cycle labels
    for z, c in zip(z_planes, CYCLES):
        ax_main.text(-1.5, 5, z, f"cycle {c}",
                      color="#1A5276", fontsize=10,
                      ha="right", va="center", fontweight="bold")
        ax_main.text(11, 0, z, f"snap_{c:02d}",
                      color="#1A5276", fontsize=8, fontweight="bold",
                      ha="left", va="center")

    # Cycle engine arrow annotation (between two planes)
    ax_main.text(5, -1.5, 1.5,
                  "↑ CycleEngine.advance()\n  (pure transition)",
                  color="#7D3C98", fontsize=9, ha="center", va="center",
                  style="italic", fontweight="bold")

    ax_main.set_xlabel("X (km)", fontsize=9)
    ax_main.set_ylabel("Y (km)", fontsize=9)
    ax_main.set_zlabel("Time →", fontsize=9)
    ax_main.set_zticks(z_planes)
    ax_main.set_zticklabels([f"c={c}" for c in CYCLES])
    ax_main.set_xlim(0, 13)
    ax_main.set_ylim(-1, 11)
    ax_main.view_init(elev=22, azim=-50)
    ax_main.set_title("Society Model — Cycle Snapshots Stacked Over Time",
                       fontsize=12, fontweight="bold", pad=10)

    # --- Inset 1: Glyph anatomy (top right) ---
    ax_glyph = fig.add_subplot(gs[0, 2:4])
    ax_glyph.set_xlim(0, 6)
    ax_glyph.set_ylim(0, 4)
    ax_glyph.axis("off")
    ax_glyph.set_title("Member glyph anatomy", fontsize=10, fontweight="bold",
                        loc="left")
    # Big sample
    ax_glyph.scatter([2.5], [2.0], s=400,
                      c=SKILL_COLORS["electrical"], marker="D",
                      edgecolors="black", linewidths=1.5, alpha=0.85)
    halo = Circle((2.5, 2.0), 0.7, facecolor="none",
                   edgecolor=HOUSEHOLD_COLORS["hh_e1"],
                   linewidth=2.2, linestyle="--")
    ax_glyph.add_patch(halo)
    # Annotations
    ax_glyph.annotate("size = energy\ncapacity (age)",
                       xy=(2.7, 2.3), xytext=(4.0, 3.4),
                       fontsize=8, ha="left",
                       arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
    ax_glyph.annotate("color = primary skill",
                       xy=(2.4, 2.2), xytext=(0.0, 3.5),
                       fontsize=8, ha="left",
                       arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
    ax_glyph.annotate("shape = vehicle\n(◇=van)",
                       xy=(2.5, 1.7), xytext=(0.0, 0.2),
                       fontsize=8, ha="left",
                       arrowprops=dict(arrowstyle="->", color="black", lw=0.7))
    ax_glyph.annotate("halo = household",
                       xy=(3.2, 2.4), xytext=(4.0, 0.4),
                       fontsize=8, ha="left",
                       arrowprops=dict(arrowstyle="->", color="black", lw=0.7))

    # --- Inset 2: Cycle engine flow (middle right) ---
    ax_eng = fig.add_subplot(gs[1, 2:4])
    ax_eng.set_xlim(0, 10)
    ax_eng.set_ylim(0, 10)
    ax_eng.axis("off")
    ax_eng.set_title("CycleEngine.advance(): pure state transition",
                      fontsize=10, fontweight="bold", loc="left")
    steps = [
        (0.3, 8.5, "Snapshot N\n(frozen)", "#D4E6F1"),
        (0.3, 7.0, "Credit transfers", "#FAD7A0"),
        (0.3, 5.7, "Household pool (30%)", "#FAD7A0"),
        (0.3, 4.4, "Reputation Δ", "#FAD7A0"),
        (0.3, 3.1, "Demurrage", "#FAD7A0"),
        (0.3, 1.8, "Energy recovery\n(age-based)", "#FAD7A0"),
        (0.3, 0.2, "Snapshot N+1\n→ snapshots.jsonl", "#ABEBC6"),
    ]
    for x, y, label, color in steps:
        ax_eng.add_patch(FancyBboxPatch((x, y - 0.2), 9.2, 0.95,
                                         boxstyle="round,pad=0.04",
                                         facecolor=color, edgecolor="black", lw=0.7))
        ax_eng.text(x + 4.6, y + 0.27, label, ha="center", va="center",
                     fontsize=8, fontweight="bold")

    # --- Inset 3: Color/shape legend (bottom right) ---
    ax_leg = fig.add_subplot(gs[2, 2:4])
    ax_leg.set_xlim(0, 10)
    ax_leg.set_ylim(0, 10)
    ax_leg.axis("off")
    ax_leg.set_title("Encoding legend", fontsize=10, fontweight="bold",
                      loc="left")
    skill_items = list(SKILL_COLORS.items())
    for i, (skill, color) in enumerate(skill_items):
        ax_leg.scatter([0.7], [8.5 - i * 0.95], s=180, c=color,
                        marker="o", edgecolors="black", linewidths=0.7)
        ax_leg.text(1.4, 8.5 - i * 0.95, skill.capitalize(),
                     fontsize=9, va="center")
    veh_items = [("none", "○"), ("bike", "□"), ("car", "△"), ("van", "◇")]
    for i, (veh, sym) in enumerate(veh_items):
        ax_leg.scatter([5.5], [8.5 - i * 0.95], s=180,
                        c="gray", marker=VEHICLE_MARKER[veh],
                        edgecolors="black", linewidths=0.7)
        ax_leg.text(6.2, 8.5 - i * 0.95, f"{sym} {veh}",
                     fontsize=9, va="center")

    # Trajectory color legend
    ax_leg.plot([0.5, 1.2], [0.7, 0.7], color="#27AE60", lw=2)
    ax_leg.text(1.4, 0.7, "growing balance", fontsize=8, va="center")
    ax_leg.plot([5.0, 5.7], [0.7, 0.7], color="#7F8C8D", lw=2)
    ax_leg.text(5.9, 0.7, "stable", fontsize=8, va="center")
    ax_leg.plot([0.5, 1.2], [-0.3, -0.3], color="#E74C3C", lw=2)
    ax_leg.text(1.4, -0.3, "demurrage erosion", fontsize=8, va="center")

    fig.suptitle("OrquestIA Society Model — Members in Space, Evolving Through Time",
                  fontsize=14, fontweight="bold", y=0.97)

    fig.savefig(os.path.join(HERE, "society_optionC_hybrid.png"),
                 bbox_inches="tight")
    plt.close(fig)
    print("  wrote society_optionC_hybrid.png")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    print("Generating society visualization...")
    option_b_multipanel()  # → fig4_society.png (publication figure)
    if "--all" in sys.argv:
        option_a_rich_3d()
        option_c_hybrid()
    print("Done. Saved to", HERE)
