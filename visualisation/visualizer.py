"""
visualisation/visualizer.py
---------------------------
Minimalist academic UI inspired by Fig.4 of Facinelli/Fontanelli 2019.

Design principles:
  * One accent colour per mode (no rainbow)
  * Whitespace over dividers
  * Small, light typography
  * Map dominates; right panel is a quiet sidebar
  * Bottom bar is a hairline strip of controls
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import config
from simulation import Simulation
from scenario   import ScenarioManager
from agents     import Stage, STAGE_NAMES

# ── Minimalist palette ──────────────────────────────────────────────────────
BG     = "#ffffff"
PANEL  = "#ffffff"
HAIR   = "#e0e2e8"
TEXT   = "#1a1d29"
DIM    = "#9097a8"
MUTED  = "#cfd2db"

ACCENT_LOC  = "#2a6280"   # deep teal
ACCENT_DEG  = "#a06820"   # bronze
ACCENT_FORM = "#3a7050"   # forest

OK     = "#3a7050"
WARN   = "#a06820"
DANGER = "#9c3030"

GAS_CM = LinearSegmentedColormap.from_list(
    "gas", ["#fffff5", "#f5edd0", "#d4b070", "#a06830", "#603018"])


class Visualizer:

    def __init__(self, sim: Simulation):
        self.sim       = sim
        self.paused    = False
        self.speed_mul = 1
        self._buttons  = []
        self._build_figure()

    # ── Figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8.5), facecolor=BG)
        try:
            self.fig.canvas.manager.set_window_title(
                "Distributed Perception")
        except Exception:
            pass

        self.ax_map = self.fig.add_axes([0.035, 0.105, 0.60, 0.84])

        self.ax_panel = self.fig.add_axes([0.665, 0.085, 0.32, 0.86])
        self.ax_panel.set_facecolor(BG)
        self.ax_panel.set_xticks([]); self.ax_panel.set_yticks([])
        self.ax_panel.set_xlim(0, 1); self.ax_panel.set_ylim(0, 1)
        for sp in self.ax_panel.spines.values():
            sp.set_visible(False)

        self.ax_ctrl = self.fig.add_axes([0.035, 0.008, 0.955, 0.055])
        self.ax_ctrl.set_facecolor(BG)
        self.ax_ctrl.set_xticks([]); self.ax_ctrl.set_yticks([])
        self.ax_ctrl.set_xlim(0, 1); self.ax_ctrl.set_ylim(0, 1)
        for sp in self.ax_ctrl.spines.values():
            sp.set_visible(False)

        self.fig.canvas.mpl_connect("key_press_event",    self._key)
        self.fig.canvas.mpl_connect("button_press_event", self._click)

    def _accent(self):
        return {ScenarioManager.LOCALISATION: ACCENT_LOC,
                ScenarioManager.DEGRADATION:  ACCENT_DEG,
                ScenarioManager.FORMATION:    ACCENT_FORM}[
                    self.sim.scenario.active_key]

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _key(self, e):
        step = self.sim.step_count
        k    = e.key
        if k == " ":           self.paused = not self.paused
        elif k == "r":          self.sim = Simulation()
        elif k in ("q","escape"): plt.close("all"); sys.exit(0)
        elif k == "l":
            m = self.sim.scenario.active
            if hasattr(m, "force_new_leak"): m.force_new_leak(step)
        elif k == "k": self.sim.kill_random_robot()
        elif k == "d":
            if self.sim.scenario.active_key == ScenarioManager.DEGRADATION:
                self.sim.scenario.active.degrade_random(step)
        elif k == "a":
            if self.sim.scenario.active_key == ScenarioManager.DEGRADATION:
                self.sim.scenario.active.toggle_auto(step)
        elif k in [str(i) for i in range(5)]:
            if self.sim.scenario.active_key == ScenarioManager.DEGRADATION:
                rid = int(k)
                if rid < config.NUM_ROBOTS:
                    self.sim.scenario.active.degrade_specific(step, rid)
        elif k == "f":
            if self.sim.scenario.active_key == ScenarioManager.FORMATION:
                self.sim.scenario.active.cycle_shape(step)
        elif k == "c" and self.sim.scenario.active_key == ScenarioManager.FORMATION:
            self.sim.scenario.active.set_shape(step, "circle")
        elif k == "n" and self.sim.scenario.active_key == ScenarioManager.FORMATION:
            self.sim.scenario.active.set_shape(step, "line")
        elif k == "v" and self.sim.scenario.active_key == ScenarioManager.FORMATION:
            self.sim.scenario.active.set_shape(step, "v")
        elif k == "t" and self.sim.scenario.active_key == ScenarioManager.FORMATION:
            self.sim.scenario.active.set_shape(step, "triangle")
        elif k in ("+","="): self.speed_mul = min(self.speed_mul*2, 16)
        elif k == "-":        self.speed_mul = max(self.speed_mul//2, 1)

    def _click(self, e):
        if e.xdata is None or e.ydata is None: return
        for ax, x0, y0, x1, y1, cb in self._buttons:
            if e.inaxes is ax and x0 <= e.xdata <= x1 and y0 <= e.ydata <= y1:
                cb(); return

    # ── Map drawing ───────────────────────────────────────────────────────────

    def _draw_map(self):
        ax  = self.ax_map
        sim = self.sim
        ax.clear()

        W, H = config.MAP_W, config.MAP_H
        scale = 0.1
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_facecolor("#fcfcfa")
        ax.tick_params(labelsize=7, colors=DIM, length=2)
        xticks = np.arange(0, W+1, 100); yticks = np.arange(0, H+1, 100)
        ax.set_xticks(xticks); ax.set_yticks(yticks)
        ax.set_xticklabels([f"{int(x*scale)}" for x in xticks])
        ax.set_yticklabels([f"{int(y*scale)}" for y in yticks])
        ax.set_xlabel("X [m]", fontsize=8, color=DIM, labelpad=2)
        ax.set_ylabel("Y [m]", fontsize=8, color=DIM, labelpad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor(HAIR); sp.set_linewidth(0.8)

        for x in xticks: ax.axvline(x, color="#f0f0ec", lw=0.4, zorder=0)
        for y in yticks: ax.axhline(y, color="#f0f0ec", lw=0.4, zorder=0)

        ph    = sim.scenario.phase
        alive = [r for r in sim.robots if r.is_alive]
        dead  = [r for r in sim.robots if not r.is_alive]

        if sim.gas.grid.max() > 0.01:
            ax.imshow(sim.gas.grid, extent=[0, W, 0, H],
                      origin="lower", cmap=GAS_CM, alpha=0.65,
                      vmin=0, vmax=1, zorder=1,
                      interpolation="bilinear", aspect=None)

        if sim.obstacles:
            for (x0, y0, x1, y1) in sim.obstacles:
                ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                    facecolor="#e6e8ec", edgecolor=HAIR, lw=0.6, zorder=3))

        # Voronoi cells — always visible to show robots' regions of
        # responsibility. Slightly more transparent during LEAK/CONSENSUS
        # so the consensus markers remain readable.
        if sim.scenario.active_key in (ScenarioManager.LOCALISATION,
                                       ScenarioManager.DEGRADATION):
            cell_alpha = 0.30 if "MONITOR" in ph or "REPORTED" in ph else 0.15
            for r in alive:
                cell = sim.lloyd.cells.get(r.id) if hasattr(sim.lloyd, 'cells') else None
                if cell and not cell.is_empty:
                    try:
                        polys = list(cell.geoms) if cell.geom_type != "Polygon" else [cell]
                        for poly in polys:
                            xs, ys = poly.exterior.xy
                            ax.plot(xs, ys, color=r.color, lw=0.5,
                                    alpha=cell_alpha, zorder=2)
                    except Exception:
                        pass

        if sim.comm_graph.A is not None:
            id2r = {r.id: r for r in alive}
            for i_id, j_id in sim.comm_graph.edges():
                ri = id2r.get(i_id); rj = id2r.get(j_id)
                if ri and rj:
                    active_link = ri.gossip_updated or rj.gossip_updated
                    ax.plot([ri.position[0], rj.position[0]],
                            [ri.position[1], rj.position[1]],
                            color=self._accent() if active_link else MUTED,
                            alpha=0.45 if active_link else 0.18,
                            lw=0.8, zorder=4)

        for r in alive:
            ax.add_patch(Circle(r.position, r.sense_r,
                color=r.color, fill=False, lw=0.6, ls=(0,(2,3)),
                alpha=0.45, zorder=5))

        for r in alive:
            trail = list(r.trail)
            n = len(trail)
            for k in range(1, n):
                alpha = (k/n) * 0.5
                ax.plot([trail[k-1][0], trail[k][0]],
                        [trail[k-1][1], trail[k][1]],
                        color=r.color, alpha=alpha, lw=0.9, zorder=6)

        if "CONSENSUS" in ph or "REPORTED" in ph:
            if "CONSENSUS" in ph:
                for r in alive:
                    if r.dkf_mu is not None:
                        ax.plot(*r.dkf_mu, marker="x", color=r.color,
                                ms=10, mew=1.8, alpha=0.85, zorder=14)
            net_mu, _ = sim.dkf.network_belief()
            if net_mu is not None:
                ax.plot(*net_mu, marker="+", color=TEXT,
                        ms=16, mew=2.0, alpha=0.95, zorder=15)
                nearest = min(alive, key=lambda r: np.linalg.norm(r.position - net_mu))
                ax.plot(*nearest.position, marker="*", color=DANGER,
                        ms=13, zorder=16, mec="white", mew=0.5)

        if "REPORTED" in ph and sim.scenario.agreed_px:
            x, y = sim.scenario.agreed_px
            ax.plot(x, y, "o", color=OK, ms=11, mew=1.8,
                    markerfacecolor="none", zorder=16)
            ax.plot(x, y, "+", color=OK, ms=14, mew=1.8, zorder=17)
            ax.annotate(
                f"$\\hat{{p}}_s$ = ({x*scale:.1f}, {y*scale:.1f}) m",
                xy=(x, y), xytext=(x+30, y+30),
                fontsize=8, color=OK,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="white", edgecolor=OK, lw=0.8, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=OK, lw=0.8),
                zorder=18)

        if sim.scenario.active_key == ScenarioManager.FORMATION:
            fc = sim.formation_ctrl
            cx, cy = fc.center
            ax.plot(cx, cy, "+", color=DIM, ms=12, mew=1.0, alpha=0.4, zorder=10)
            n = sum(1 for r in alive)
            for r in alive:
                if r.formation_slot is None: continue
                off = fc.slot_offset(r.formation_slot, n)
                tx, ty = cx+off[0], cy+off[1]
                ax.plot(tx, ty, "s", color=r.color, ms=6, alpha=0.25,
                        markerfacecolor="none", mew=0.8, zorder=9)
                ax.plot([r.position[0], tx], [r.position[1], ty],
                        color=r.color, lw=0.4, alpha=0.35, ls=(0,(2,3)),
                        zorder=8)

        for r in alive:
            pos = r.position
            ring = {Stage.HEALTHY: r.color, Stage.NOISY: WARN,
                    Stage.FAILING: DANGER, Stage.DEAD: MUTED}[r.stage]
            ax.add_patch(Circle(pos, 8, facecolor=r.color,
                edgecolor="white", lw=1.2, zorder=12))
            if r.stage != Stage.HEALTHY:
                ax.add_patch(Circle(pos, 11, facecolor="none",
                    edgecolor=ring, lw=1.4, zorder=13))
            if r.manual_degradation:
                ax.add_patch(Circle(pos, 15, facecolor="none",
                    edgecolor=DANGER, lw=0.8, ls=(0,(2,2)), zorder=13))
            ax.text(pos[0], pos[1]+14, f"R{r.id}",
                fontsize=6.5, ha="center", va="bottom",
                color=DIM, zorder=14)

        for r in dead:
            ax.plot(*r.position, "x", color=DANGER, ms=9, mew=2, zorder=12)

        if sim.scenario.source_pxs and "REPORTED" in ph:
            for src in sim.scenario.source_pxs:
                ax.plot(*src, "^", color="#cc2200", ms=8, mew=0,
                        alpha=0.65, zorder=15)

        legend_elements = [
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor=config.PALETTE[0], ms=7, label="robot"),
            Line2D([0],[0], marker="x", color=DIM, ms=8, mew=1.5,
                   label="belief"),
            Line2D([0],[0], marker="+", color=TEXT, ms=10, mew=1.8,
                   label="consensus"),
            Line2D([0],[0], marker="*", color=DANGER, ms=8, label="leader"),
        ]
        ax.legend(handles=legend_elements, loc="lower right",
                  fontsize=6.5, framealpha=0.85, facecolor="white",
                  edgecolor=HAIR, labelcolor=DIM, ncol=4,
                  handletextpad=0.2, columnspacing=0.8,
                  bbox_to_anchor=(0.998, 0.005))

    # ── Right sidebar ─────────────────────────────────────────────────────────

    def _draw_panel(self):
        ax  = self.ax_panel
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp in ax.spines.values(): sp.set_visible(False)

        sim = self.sim
        acc = self._accent()
        y = 0.97

        # ── PHASE CHIP ──────────────────────────────────────────────────────
        phase  = sim.scenario.phase.split("—")[0].strip().upper()
        ax.add_patch(FancyBboxPatch((0, y-0.04), 0.85, 0.038,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor=acc, edgecolor="none",
            transform=ax.transAxes, zorder=2))
        ax.text(0.025, y-0.021, phase,
            color="white", fontsize=9, fontweight="bold",
            transform=ax.transAxes, va="center")
        mode_num = sim.scenario.active_key
        ax.text(0.95, y-0.021, f"·{mode_num}",
            color="white", fontsize=10, fontweight="bold",
            transform=ax.transAxes, va="center", ha="right")
        y -= 0.075

        # ── METRICS ──────────────────────────────────────────────────────────
        lam2 = sim.scenario.lambda2_history[-1] if sim.scenario.lambda2_history else 0.0
        components = sim.comm_graph.num_components() if sim.comm_graph.A is not None else 1
        lam_col = OK if lam2 > 0.5 else (WARN if lam2 > 0.05 else DANGER)
        conn_str = "connected" if components <= 1 else f"{components} parts"

        # σ (belief spread) is only meaningful when robots are actively
        # building/refining beliefs about a leak — during CONSENSUS and
        # for a brief moment during REPORTED. In MONITORING / LEAK there
        # are no beliefs to compare, so showing a stale value is
        # misleading. Display "—" instead.
        ph_upper = sim.scenario.phase.upper()
        if ("CONSENSUS" in ph_upper or "REPORTED" in ph_upper) \
           and sim.scenario.consensus_history:
            sigma = sim.scenario.consensus_history[-1]
            sigma_str = (f"{sigma:.1f} px"
                         if sigma is not None and sigma < 900
                         else "—")
        else:
            sigma_str = "—"

        metrics = [
            ("step",        f"{sim.step_count}",                 TEXT, ""),
            ("alive",       f"{sim.alive_count}/{len(sim.robots)}", TEXT, ""),
            ("leaks",       f"{sim.scenario.leak_count}",        TEXT, ""),
            ("λ₂",          f"{lam2:.2f}",                       lam_col, conn_str),
            ("σ",           sigma_str,                            TEXT, "spread"),
        ]
        for label, val, vcol, hint in metrics:
            ax.text(0.0, y, label,
                color=DIM, fontsize=8,
                transform=ax.transAxes, va="top", family="monospace")
            ax.text(0.30, y, val,
                color=vcol, fontsize=9,
                transform=ax.transAxes, va="top", family="monospace")
            if hint:
                ax.text(0.96, y, hint,
                    color=DIM, fontsize=7,
                    transform=ax.transAxes, va="top", ha="right")
            y -= 0.030

        y -= 0.020

        # ── ROBOTS ────────────────────────────────────────────────────────────
        ax.text(0.0, y, "ROBOTS",
            color=DIM, fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top", family="monospace")
        y -= 0.025

        show_trust = (sim.scenario.active_key == ScenarioManager.DEGRADATION)

        for r in sim.robots:
            col = TEXT if r.is_alive else DIM
            ax.add_patch(Circle((0.025, y-0.011), 0.011,
                facecolor=r.color if r.is_alive else MUTED,
                edgecolor="none", transform=ax.transAxes, zorder=3))
            ax.text(0.065, y, f"R{r.id}",
                color=col, fontsize=8,
                transform=ax.transAxes, va="top", family="monospace")
            stage_str = STAGE_NAMES[r.stage]
            if r.manual_degradation and r.stage != Stage.DEAD:
                stage_str += "*"
            stage_col = OK if r.stage == Stage.HEALTHY else (
                WARN if r.stage == Stage.NOISY else DANGER)
            ax.text(0.135, y, stage_str,
                color=stage_col, fontsize=8,
                transform=ax.transAxes, va="top", family="monospace")

            if show_trust:
                ax.text(0.55, y, f"τ={r.own_reputation:.2f}",
                    color=DIM, fontsize=7.5,
                    transform=ax.transAxes, va="top",
                    family="monospace")
            else:
                obs_mark = "●" if r.self_observed else "○"
                ax.text(0.55, y, obs_mark,
                    color=acc if r.self_observed else MUTED, fontsize=9,
                    transform=ax.transAxes, va="top")
            y -= 0.026

        y -= 0.020

        # ── EVENT LOG ─────────────────────────────────────────────────────────
        ax.text(0.0, y, "LOG",
            color=DIM, fontsize=7, fontweight="bold",
            transform=ax.transAxes, va="top", family="monospace")
        y -= 0.022

        events = list(sim.log.events)[-12:][::-1]
        for ev in events:
            sev_col = {"info": TEXT, "event": acc,
                       "warn": WARN, "critical": DANGER}.get(ev["sev"], TEXT)
            t_str  = f"{ev['step']:>4}"
            msg    = ev['msg']
            tag    = ev['tag']
            # Truncate long messages but keep enough to read key fields
            if len(msg) > 42: msg = msg[:41] + "…"
            ax.text(0.0,  y, t_str, color=DIM, fontsize=7,
                transform=ax.transAxes, va="top", family="monospace")
            ax.text(0.10, y, tag,   color=DIM, fontsize=7,
                transform=ax.transAxes, va="top", family="monospace")
            ax.text(0.30, y, msg,   color=sev_col, fontsize=6.5,
                transform=ax.transAxes, va="top", family="monospace")
            y -= 0.020
            if y < 0.02: break

    # ── Bottom control strip ──────────────────────────────────────────────────

    def _draw_controls(self):
        ax = self.ax_ctrl
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for sp in ax.spines.values(): sp.set_visible(False)

        # Top hairline
        ax.axhline(0.97, color=HAIR, lw=0.6, xmin=0, xmax=1)

        # Mode pills
        modes = [
            (ScenarioManager.LOCALISATION, "Localization",   ACCENT_LOC),
            (ScenarioManager.DEGRADATION,  "Degradation",    ACCENT_DEG),
            (ScenarioManager.FORMATION,    "Formation",      ACCENT_FORM),
        ]
        for i, (key, label, col) in enumerate(modes):
            x0 = 0.005 + i * 0.115
            w  = 0.105
            active = (key == self.sim.scenario.active_key)
            ax.add_patch(FancyBboxPatch((x0, 0.30), w, 0.45,
                boxstyle="round,pad=0.005,rounding_size=0.04",
                facecolor=col if active else "white",
                edgecolor=col if active else HAIR,
                lw=0.8, transform=ax.transAxes))
            ax.text(x0+w/2+0.005, 0.525, label,
                color="white" if active else DIM,
                fontsize=8.5,
                fontweight="bold" if active else "normal",
                ha="center", va="center", transform=ax.transAxes)
            ax.text(x0+0.012, 0.525, f"{key}",
                color="white" if active else col,
                fontsize=7, fontweight="bold",
                ha="left", va="center", transform=ax.transAxes)

            def mk_cb(k=key):
                def cb(): self.sim.scenario.switch_to(k, self.sim.step_count)
                return cb
            self._buttons.append((ax, x0, 0.20, x0+w, 0.85, mk_cb()))

        # Speed pills
        ax.text(0.395, 0.525, "speed",
            color=DIM, fontsize=7.5,
            ha="right", va="center", transform=ax.transAxes,
            family="monospace")
        for i, m in enumerate([1, 2, 4, 8, 16]):
            x0 = 0.405 + i * 0.034
            w  = 0.030
            active = (m == self.speed_mul)
            ax.add_patch(FancyBboxPatch((x0, 0.32), w, 0.40,
                boxstyle="round,pad=0.003,rounding_size=0.03",
                facecolor=TEXT if active else "white",
                edgecolor=TEXT if active else HAIR,
                lw=0.6, transform=ax.transAxes))
            ax.text(x0+w/2, 0.525, f"{m}×",
                color="white" if active else DIM,
                fontsize=7.5,
                fontweight="bold" if active else "normal",
                ha="center", va="center", transform=ax.transAxes)
            def mk_speed(mm=m):
                def cb(): self.speed_mul = mm
                return cb
            self._buttons.append((ax, x0, 0.20, x0+w, 0.85, mk_speed()))

        # Hint
        active_key = self.sim.scenario.active_key
        if active_key == ScenarioManager.LOCALISATION:
            hint = "L leak  ·  K kill  ·  Space pause  ·  R reset  ·  Q quit"
        elif active_key == ScenarioManager.DEGRADATION:
            hint = "L leak  ·  D degrade  ·  0–4 robot  ·  A auto  ·  K kill"
        else:
            hint = "C circle  ·  N line  ·  V v-shape  ·  T triangle  ·  F cycle"
        ax.text(0.60, 0.525, hint,
            color=DIM, fontsize=7.5,
            ha="left", va="center", transform=ax.transAxes,
            family="monospace")

        if self.paused:
            ax.text(0.999, 0.525, "PAUSED",
                color=WARN, fontsize=8, fontweight="bold",
                ha="right", va="center", transform=ax.transAxes,
                family="monospace")

    # ── Animation ─────────────────────────────────────────────────────────────

    def _frame(self, _):
        self._buttons = []
        if not self.paused:
            for _ in range(self.speed_mul):
                self.sim.step()
        self._draw_map()
        self._draw_panel()
        self._draw_controls()
        return []

    def run(self):
        self._ani = animation.FuncAnimation(
            self.fig, self._frame,
            interval=1000//config.FPS,
            blit=False, cache_frame_data=False)
        plt.show()
