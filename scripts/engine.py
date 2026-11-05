"""
Universal Matplotlib Architecture & Flowchart Engine.
Supports arbitrary canvas sizes, flexible aspect ratios, customizable color themes,
collision-aware automatic placement of labels and legends,
and publication-grade vector SVG / high-res PNG outputs.
"""

import os
import warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

from .color_palettes import get_theme, ACADEMIC_LIGHT

BACKGROUND_MODES = ("theme", "trans")

# Obstacle kinds used by the auto-placement engine
ALL_KINDS = ("box", "text", "line", "border")
CONTENT_KINDS = ("box", "text", "line")


# ----------------------------------------------------------------------------
# Small geometry helpers (all in data coordinates)
# ----------------------------------------------------------------------------
def _rects_overlap(a, b, pad=0.0):
    """True if rects (x0, y0, x1, y1) overlap or are closer than `pad`."""
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or
                a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _seg_hits_rect(seg, rect):
    """Liang-Barsky segment vs. axis-aligned rect intersection test."""
    x1, y1, x2, y2 = seg[:4]
    dx, dy = x2 - x1, y2 - y1
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - rect[0]), (dx, rect[2] - x1),
                 (-dy, y1 - rect[1]), (dy, rect[3] - y1)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


class FlowchartCanvas:
    """
    Universal Canvas for programmatically drawing publication-quality flowcharts,
    neural network modules, and pipeline diagrams.

    Auto-placement: cards, pills, nodes, arrows, stage titles and stage borders are
    remembered as obstacles. `label()`, `path_label()` and `legend()` (without x/y)
    are resolved when `layout()` runs (automatically inside `save()`), so they are
    placed in free space regardless of the order in which things were drawn.
    """
    def __init__(self, figsize=(16.0, 5.5), xlim=(0, 100), ylim=(0, 33),
                 aspect="equal", theme="academic", canvas_bg=None, background="theme"):
        """
        background:
            "theme" (default) - solid canvas color taken from the theme
                                (white for light themes, dark gray for "dark").
            "trans"           - fully transparent background (only the shapes are drawn).
        canvas_bg:
            Optional hex color overriding the theme canvas color (ignored when background="trans").
        """
        if background not in BACKGROUND_MODES:
            raise ValueError(f"background must be one of {BACKGROUND_MODES}, got {background!r}")
        self.background = background
        self.theme = get_theme(theme)
        self.figsize = figsize
        self.xlim = xlim
        self.ylim = ylim
        self.aspect = aspect

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        if aspect:
            self.ax.set_aspect(aspect)
        self.ax.axis("off")

        if background == "theme":
            bg = canvas_bg or self.theme.get("canvas_bg", "#FFFFFF")
            self.fig.patch.set_facecolor(bg)
            self.ax.set_facecolor(bg)

        # Obstacle registry for auto-placement
        self._boxes = []        # (x0, y0, x1, y1): cards, pills, nodes, placed labels, legends
        self._texts = []        # tracked Text artists (measured at layout time)
        self._text_rects = []   # measured extents of self._texts
        self._lines = []        # (x1, y1, x2, y2, group): arrow / path segments
        self._borders = []      # (x1, y1, x2, y2): stage container outlines
        self._pending = []      # deferred placement callbacks
        self._renderer = None

    @classmethod
    def wide(cls, figsize=(16.0, 5.5), xlim=(0, 100), ylim=(0, 33), **kwargs):
        """Standard wide banner layout (great for horizontal multi-stage pipelines)."""
        return cls(figsize=figsize, xlim=xlim, ylim=ylim, **kwargs)

    @classmethod
    def tall(cls, figsize=(9.0, 12.0), xlim=(0, 60), ylim=(0, 100), **kwargs):
        """Vertical stack layout (great for deep Transformers, Mamba blocks, ResNets)."""
        return cls(figsize=figsize, xlim=xlim, ylim=ylim, **kwargs)

    @classmethod
    def square(cls, figsize=(8.0, 8.0), xlim=(0, 50), ylim=(0, 50), **kwargs):
        """Square layout (great for standalone components or single-module deep-dives)."""
        return cls(figsize=figsize, xlim=xlim, ylim=ylim, **kwargs)

    def get_color(self, color_key_or_hex, role="border"):
        """Resolves color from theme palette or returns direct hex string."""
        if not color_key_or_hex:
            return "#000000"
        if color_key_or_hex.startswith("#"):
            return color_key_or_hex
        pal = self.theme.get(color_key_or_hex, self.theme.get("neutral", {}))
        return pal.get(role, "#000000")

    # ------------------------------------------------------------------
    # Obstacle bookkeeping + auto-placement core
    # ------------------------------------------------------------------
    def _add_box(self, x, y, w, h, extra=0.0):
        """Registers a rectangular obstacle (extra = shadow / stack offset toward lower right)."""
        self._boxes.append((x, y - extra, x + w + extra, y + h))

    def text(self, x, y, s, **kwargs):
        """Draws centered text (like ax.text) and tracks it as an obstacle for auto-placement."""
        kwargs.setdefault("ha", "center")
        kwargs.setdefault("va", "center")
        t = self.ax.text(x, y, s, **kwargs)
        self._texts.append(t)
        return t

    def _text_rect(self, t):
        """Current data-space extent of a Text artist (needs a prepared renderer)."""
        bb = t.get_window_extent(self._renderer)
        (x0, y0), (x1, y1) = self.ax.transData.inverted().transform(
            [(bb.x0, bb.y0), (bb.x1, bb.y1)])
        return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def _conflicts(self, rect, kinds=CONTENT_KINDS, ignore_group=None, pad=0.25, first_only=False):
        """Counts obstacles that collide with `rect` (stops at the first one if first_only)."""
        n = 0
        if "box" in kinds:
            for b in self._boxes:
                if _rects_overlap(rect, b, pad):
                    n += 1
                    if first_only:
                        return n
        if "text" in kinds:
            for b in self._text_rects:
                if _rects_overlap(rect, b, pad):
                    n += 1
                    if first_only:
                        return n
        padded = (rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad)
        if "line" in kinds:
            for s in self._lines:
                if ignore_group is not None and s[4] == ignore_group:
                    continue
                if _seg_hits_rect(s, padded):
                    n += 1
                    if first_only:
                        return n
        if "border" in kinds:
            for s in self._borders:
                if _seg_hits_rect(s, padded):
                    n += 1
                    if first_only:
                        return n
        return n

    def _inside_canvas(self, rect, margin=0.3):
        return (rect[0] >= self.xlim[0] + margin and rect[2] <= self.xlim[1] - margin and
                rect[1] >= self.ylim[0] + margin and rect[3] <= self.ylim[1] - margin)

    def layout(self):
        """
        Resolves every deferred placement (label / path_label / auto legend).
        Called automatically by save(); call it manually only if you need the final
        positions earlier. Items are resolved in the order they were requested.
        """
        if not self._pending:
            return
        self.fig.tight_layout(pad=0.05)
        self.fig.canvas.draw()
        self._renderer = self.fig.canvas.get_renderer()
        self._text_rects = [self._text_rect(t) for t in self._texts]
        pending, self._pending = self._pending, []
        for place in pending:
            place()

    def rbox(self, x, y, w, h, fc="#FFFFFF", ec="#000000", lw=1.0, ls="-",
             rad=0.8, shadow=False, alpha=1.0, z=2, stack=0):
        """Draws a rounded rectangular box with optional drop shadow or 3D card stack."""
        if shadow:
            sb = FancyBboxPatch((x + 0.22, y - 0.22), w, h,
                                boxstyle=f"round,pad=0,rounding_size={rad}",
                                fc="#000000", ec="none", alpha=0.08, zorder=z - 1)
            self.ax.add_patch(sb)
        if stack > 0:
            for s in range(stack, 0, -1):
                offset = s * 0.28
                stk = FancyBboxPatch((x + offset, y - offset), w, h,
                                     boxstyle=f"round,pad=0,rounding_size={rad}",
                                     fc=fc, ec=ec, lw=lw, ls=ls, alpha=0.6, zorder=z)
                self.ax.add_patch(stk)
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle=f"round,pad=0,rounding_size={rad}",
                             fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z)
        self.ax.add_patch(box)
        return box

    def stage_container(self, x, y, w, h, title=None, palette="neutral",
                        ls="--", lw=1.35, rad=1.2, z=1, title_align="center"):
        """Draws a high-level stage container grouping related operations.

        title_align: "center" (default), "left" or "right" - use it to keep the title
        clear of an arrow entering/leaving the container at its horizontal center.
        """
        fc = self.get_color(palette, role="bg")
        ec = self.get_color(palette, role="border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, rad=rad, alpha=1.0, z=z)
        # Outline is an obstacle for things that must not straddle the stage border (legends)
        self._borders += [(x, y, x + w, y), (x + w, y, x + w, y + h),
                          (x + w, y + h, x, y + h), (x, y + h, x, y)]
        if title:
            tx, ha = {"left": (x + 1.5, "left"), "right": (x + w - 1.5, "right")}.get(
                title_align, (x + w / 2, "center"))
            self.text(tx, y + h - 1.5, title, ha=ha, fontsize=10.5, fontweight="bold",
                      color=tc, zorder=z + 2)
        return box

    def card(self, x, y, w, h, title, subtitle=None, formula=None, spec=None,
             palette="neutral", shadow=True, stack=0, rad=0.6, lw=0.95, z=3):
        """Draws an operation card with balanced typography and optional shadow/stack."""
        fc = self.get_color(palette, role="card_bg")
        ec = self.get_color(palette, role="card_border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=lw, rad=rad, shadow=shadow, stack=stack, z=z)
        self._add_box(x, y, w, h, extra=max(0.22 if shadow else 0.0, stack * 0.28))

        # Distribute text lines evenly
        lines = []
        if title:
            lines.append({"text": title, "size": 8.2, "weight": "bold", "color": tc})
        if formula:
            lines.append({"text": formula, "size": 7.6, "weight": "bold", "color": "#212529"})
        if subtitle:
            lines.append({"text": subtitle, "size": 6.8, "weight": "normal", "color": "#555555"})
        if spec:
            lines.append({"text": spec, "size": 6.5, "weight": "normal", "color": tc})

        n = len(lines)
        if n == 1:
            self.ax.text(x + w / 2, y + h / 2, lines[0]["text"], ha="center", va="center",
                         fontsize=lines[0]["size"], fontweight=lines[0]["weight"],
                         color=lines[0]["color"], zorder=z + 2)
        elif n == 2:
            self.ax.text(x + w / 2, y + h / 2 + 0.65, lines[0]["text"], ha="center", va="center",
                         fontsize=lines[0]["size"], fontweight=lines[0]["weight"],
                         color=lines[0]["color"], zorder=z + 2)
            self.ax.text(x + w / 2, y + h / 2 - 0.65, lines[1]["text"], ha="center", va="center",
                         fontsize=lines[1]["size"], fontweight=lines[1]["weight"],
                         color=lines[1]["color"], zorder=z + 2)
        elif n >= 3:
            spacing = (h - 2.0) / (n - 1) if n > 1 else 0
            for idx, l in enumerate(lines):
                cy = (y + h - 1.0) - idx * spacing
                self.ax.text(x + w / 2, cy, l["text"], ha="center", va="center",
                             fontsize=l["size"], fontweight=l["weight"], color=l["color"], zorder=z + 2)
        return box

    def pill(self, x, y, w, h, label, badge=None, palette="blue", rad=0.8, shadow=True, z=3):
        """Draws a compact feature pill with an optional badge chip (e.g. dimensions)."""
        fc = self.get_color(palette, role="tint")
        ec = self.get_color(palette, role="border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=0.9, rad=rad, shadow=shadow, z=z)
        self._add_box(x, y, w, h, extra=0.22 if shadow else 0.0)
        if badge:
            # text on left, badge on right
            self.ax.text(x + 1.2, y + h / 2, label, ha="left", va="center",
                         fontsize=7.4, fontweight="bold", color="#212529", zorder=z + 2)
            # badge chip
            bw, bh = 2.4, h * 0.75
            bx, by = x + w - bw - 0.6, y + (h - bh) / 2
            self.rbox(bx, by, bw, bh, fc="#FFFFFF", ec=ec, lw=0.6, rad=rad * 0.6, z=z + 1)
            self.ax.text(bx + bw / 2, by + bh / 2, str(badge), ha="center", va="center",
                         fontsize=6.8, fontweight="bold", color=tc, zorder=z + 2)
        else:
            self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                         fontsize=7.4, fontweight="bold", color="#212529", zorder=z + 2)
        return box

    def vertical_block(self, x, y, w, h, label, palette="purple", rad=0.8, rotation=90, z=3):
        """Draws a vertical backbone or encoder spine with rotated label."""
        fc = self.get_color(palette, role="card_bg")
        ec = self.get_color(palette, role="card_border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=1.0, rad=rad, shadow=True, z=z)
        self._add_box(x, y, w, h, extra=0.22)
        self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                     rotation=rotation, fontsize=8.6, fontweight="bold", color=tc, zorder=z + 2)
        return box

    def arrow(self, x1, y1, x2, y2, ls="-", col="#000000", lw=1.0, hl=0.35, hw=0.22, z=4,
              group=None):
        """Draws a straight arrow with proportional vector arrowhead."""
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            return
        self._lines.append((x1, y1, x2, y2, group))
        ux, uy = dx / length, dy / length
        self.ax.plot([x1, x2 - ux * hl], [y1, y2 - uy * hl],
                     linestyle=ls, color=col, lw=lw, zorder=z, solid_capstyle="round")
        self.ax.annotate("", xy=(x2, y2), xytext=(x2 - ux * hl, y2 - uy * hl),
                         arrowprops=dict(arrowstyle=f"-|>,head_length={hl*2.8},head_width={hw*2.8}",
                                         color=col, lw=lw),
                         zorder=z)

    def poly_arrow(self, pts, ls="-", col="#000000", lw=1.0, hl=0.35, hw=0.22, z=4):
        """Draws an orthogonal multi-segment path with an arrowhead on the final segment."""
        group = tuple(tuple(p) for p in pts)
        for i in range(len(pts) - 2):
            self._lines.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], group))
            self.ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                         linestyle=ls, color=col, lw=lw, zorder=z, solid_capstyle="round")
        p_penult, p_last = pts[-2], pts[-1]
        self.arrow(p_penult[0], p_penult[1], p_last[0], p_last[1],
                   ls=ls, col=col, lw=lw, hl=hl, hw=hw, z=z, group=group)

    def plus_node(self, x, y, r=0.75, fc="#FFFFFF", ec="#000000", lw=1.0, z=6, col="#000000"):
        """Draws an element-wise addition circular node (+)."""
        c = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z)
        self.ax.add_patch(c)
        self._boxes.append((x - r, y - r, x + r, y + r))
        d = r * 0.55
        self.ax.plot([x - d, x + d], [y, y], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")
        self.ax.plot([x, x], [y - d, y + d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")

    def times_node(self, x, y, r=0.75, fc="#FFFFFF", ec="#000000", lw=1.0, z=6, col="#000000"):
        """Draws a matrix multiplication / Hadamard circular node (x)."""
        c = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z)
        self.ax.add_patch(c)
        self._boxes.append((x - r, y - r, x + r, y + r))
        d = r * 0.42
        self.ax.plot([x - d, x + d], [y - d, y + d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")
        self.ax.plot([x - d, x + d], [y + d, y - d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")

    # ------------------------------------------------------------------
    # Auto-placed annotations
    # ------------------------------------------------------------------
    def label(self, x, y, text, color=None, fontsize=7.2, fontweight="bold", gap=1.1,
              prefer=("above", "right", "left", "below"), max_shift=12.0, boxed=False,
              fc=None, ec=None, z=8):
        """
        Text label attached to the anchor point (x, y) (e.g. a node), automatically moved
        to the nearest free spot that does not overlap cards, nodes, arrows, titles,
        stage borders or other labels.

        prefer:     sides to favor when several spots are about equally close
                    ("above", "below", "left", "right").
        gap:        distance between the anchor and the label edge.
        max_shift:  how far the label may slide away from its ideal spot.
        boxed:      draw a rounded background box behind the text.
        """
        col = color or self.get_color("neutral", "text")
        t = self.ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                         fontweight=fontweight, color=col, zorder=z + 1)

        def place():
            x0, y0, x1, y1 = self._text_rect(t)
            w, h = (x1 - x0), (y1 - y0)
            if boxed:
                w, h = w + 1.0, h + 0.8
            # Candidate centers on a grid around the anchor, ranked by how close the label
            # rect gets to the anchor (plus a small bias toward the preferred sides).
            cands = []
            offs = np.arange(-max_shift - w, max_shift + w + 1e-9, 0.5)
            offs_y = np.arange(-max_shift - h, max_shift + h + 1e-9, 0.5)
            for dx in offs:
                for dy in offs_y:
                    cx, cy = x + dx, y + dy
                    ex = max(cx - w / 2 - x, 0.0, x - (cx + w / 2))
                    ey = max(cy - h / 2 - y, 0.0, y - (cy + h / 2))
                    d = float(np.hypot(ex, ey))
                    if d < gap or d > gap + max_shift:
                        continue
                    if abs(dy) / h >= abs(dx) / w:
                        side = "above" if dy > 0 else "below"
                    else:
                        side = "right" if dx > 0 else "left"
                    pi = prefer.index(side) if side in prefer else len(prefer)
                    cands.append((d + pi * 0.3, (cx, cy)))
            cands.sort(key=lambda e: e[0])

            best, best_n = None, None
            for _, (cx, cy) in cands:
                rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if not self._inside_canvas(rect):
                    continue
                n = self._conflicts(rect, ALL_KINDS, first_only=True)
                if n == 0:
                    best = (cx, cy, rect)
                    break
                if best_n is None:
                    best, best_n = (cx, cy, rect), n
            if best is None:
                best = (x, y + gap + h / 2, (x - w / 2, y + gap, x + w / 2, y + gap + h))
            elif best_n is not None and self._conflicts(best[2], ALL_KINDS, first_only=True):
                warnings.warn(f"label {text!r}: no free spot found, placed at best effort")
            cx, cy, rect = best
            t.set_position((cx, cy))
            if boxed:
                fcol = fc or self.get_color("neutral", "card_bg")
                self.rbox(rect[0], rect[1], w, h, fc=fcol, ec=ec or col, lw=0.75,
                          rad=0.4, z=z)
            self._boxes.append(rect)

        self._pending.append(place)
        return t

    def path_label(self, pts, text, pos=0.5, color=None, fontsize=7.2, fontweight="bold",
                   fc=None, ec=None, lw=0.75, pad=(0.6, 0.5), rad=0.4, z=8):
        """
        Boxed label sitting ON a path drawn with poly_arrow(pts). It slides along the
        path (starting at `pos`, 0..1 of the longest segment) until it no longer covers
        any card, node, title, other arrow or other label.
        """
        col = color or self.get_color("neutral", "text")
        group = tuple(tuple(p) for p in pts)
        t = self.ax.text(pts[0][0], pts[0][1], text, ha="center", va="center", fontsize=fontsize,
                         fontweight=fontweight, color=col, zorder=z + 1)

        def place():
            x0, y0, x1, y1 = self._text_rect(t)
            w, h = (x1 - x0) + 2 * pad[0], (y1 - y0) + 2 * pad[1]
            segs = []
            for i in range(len(pts) - 1):
                (ax_, ay_), (bx_, by_) = pts[i], pts[i + 1]
                segs.append((np.hypot(bx_ - ax_, by_ - ay_), (ax_, ay_), (bx_, by_)))
            segs.sort(key=lambda s: -s[0])

            best, best_n = None, None
            for L, (ax_, ay_), (bx_, by_) in segs:
                if L == 0:
                    continue
                ux, uy = (bx_ - ax_) / L, (by_ - ay_) / L
                half = (w if abs(ux) > abs(uy) else h) / 2.0
                lo, hi = half, L - half
                if hi < lo:
                    lo = hi = L / 2.0
                ss = np.arange(lo, hi + 1e-9, 0.5)
                ss = sorted(ss, key=lambda s: abs(s - pos * L))
                for s in ss:
                    cx, cy = ax_ + ux * s, ay_ + uy * s
                    rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    n = self._conflicts(rect, CONTENT_KINDS, ignore_group=group, first_only=True)
                    if n == 0:
                        best, best_n = (cx, cy, rect), 0
                        break
                    if best_n is None:
                        best, best_n = (cx, cy, rect), n
                if best_n == 0:
                    break
            if best is None:
                return
            if best_n:
                warnings.warn(f"path_label {text!r}: no free spot found, placed at best effort")
            cx, cy, rect = best
            fcol = fc or self.get_color("neutral", "card_bg")
            self.rbox(rect[0], rect[1], w, h, fc=fcol, ec=ec or col, lw=lw, rad=rad, z=z)
            t.set_position((cx, cy))
            self._boxes.append(rect)

        self._pending.append(place)
        return t

    def legend(self, x=None, y=None, items=None, title="Legend", fc="#FFFFFF", ec="#ADB5BD",
               rad=0.4, z=5, anchor="lower right"):
        """
        Draws an operation legend box with symbols and descriptions.

        With explicit x and y the legend is drawn exactly there. Leave x and/or y as None
        (e.g. `canvas.legend(items=[...])`) to let the engine pick the free rectangle
        closest to `anchor` ("lower right", "lower left", "upper right", "upper left")
        that does not touch any card, arrow, title or stage border.
        """
        if items is None and x is not None and y is None:   # legend([...]) shorthand
            x, items = None, x
        if not items:
            raise ValueError("legend() needs a non-empty items list")
        n = len(items)
        h = 2.4 + n * 1.8
        w = max(11.0, max(len(desc) for _, desc in items) * 0.65 + 4.5)

        if x is not None and y is not None:
            self._draw_legend(x, y, w, h, items, title, fc, ec, rad, z)
            return (x, y, w, h)

        def place():
            m = 0.5
            xs = [x] if x is not None else np.arange(self.xlim[0] + m, self.xlim[1] - w - m + 1e-9, 0.5)
            ys = [y] if y is not None else np.arange(self.ylim[0] + m, self.ylim[1] - h - m + 1e-9, 0.5)
            right = "right" in anchor
            top = "upper" in anchor
            cx0 = self.xlim[1] if right else self.xlim[0]
            cy0 = self.ylim[1] if top else self.ylim[0]
            cands = []
            for xx in xs:
                for yy in ys:
                    px = xx + w if right else xx
                    py = yy + h if top else yy
                    cands.append((np.hypot(px - cx0, py - cy0), xx, yy))
            cands.sort()
            best = None
            for _, xx, yy in cands:
                rect = (xx, yy, xx + w, yy + h)
                if self._conflicts(rect, ALL_KINDS, pad=0.4, first_only=True) == 0:
                    best = (xx, yy)
                    break
            if best is None:
                warnings.warn("legend: no free spot found, placed at best effort")
                scored = [(self._conflicts((xx, yy, xx + w, yy + h), ALL_KINDS, pad=0.4), d, xx, yy)
                          for d, xx, yy in cands[:3000]]
                _, _, xx, yy = min(scored)
                best = (xx, yy)
            self._draw_legend(best[0], best[1], w, h, items, title, fc, ec, rad, z)

        self._pending.append(place)
        return None

    def _draw_legend(self, x, y, w, h, items, title, fc, ec, rad, z):
        self.rbox(x, y, w, h, fc=fc, ec=ec, lw=0.8, rad=rad, z=z)
        self._boxes.append((x, y, x + w, y + h))
        self.ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="center",
                     fontsize=7.0, fontweight="bold", color="#495057", zorder=z + 1)
        for idx, (sym_type, desc) in enumerate(items):
            iy = y + h - 2.5 - idx * 1.7
            if sym_type in ["+", "plus", "sum"]:
                self.plus_node(x + 1.4, iy, r=0.45, lw=0.8, z=z + 1)
            elif sym_type in ["x", "times", "mult"]:
                self.times_node(x + 1.4, iy, r=0.45, lw=0.8, z=z + 1)
            elif sym_type in ["--", "dashed", "skip"]:
                self.ax.plot([x + 0.8, x + 2.0], [iy, iy], ls="--", color="#000000", lw=1.0, zorder=z + 1)
            self.ax.text(x + 2.6, iy, desc, ha="left", va="center",
                         fontsize=6.4, color="#212529", zorder=z + 1)

    def save(self, output_dir, name, formats=("png", "svg"), dpi=300):
        """Saves camera-ready vector SVG and high-resolution raster PNG."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.layout()
        self.fig.tight_layout(pad=0.05)
        for ext in formats:
            out_file = out_dir / f"{name}.{ext}"
            tmp_file = out_dir / f"_tmp_{name}.{ext}"
            self.fig.savefig(tmp_file, dpi=dpi, transparent=(self.background == "trans"),
                             bbox_inches="tight", pad_inches=0.04)
            tmp_file.replace(out_file)
            print(f"Saved {out_file}")
        plt.close(self.fig)
