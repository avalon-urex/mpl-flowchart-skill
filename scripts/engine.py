"""
Universal Matplotlib Architecture & Flowchart Engine.
Supports arbitrary canvas sizes, flexible aspect ratios, customizable color themes,
and publication-grade vector SVG / high-res PNG outputs.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

from .color_palettes import get_theme, ACADEMIC_LIGHT


class FlowchartCanvas:
    """
    Universal Canvas for programmatically drawing publication-quality flowcharts,
    neural network modules, and pipeline diagrams.
    """
    def __init__(self, figsize=(16.0, 5.5), xlim=(0, 100), ylim=(0, 33),
                 aspect="equal", theme="academic", canvas_bg=None):
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

        bg = canvas_bg or self.theme.get("canvas_bg", "#FFFFFF")
        if bg != "#FFFFFF":
            self.fig.patch.set_facecolor(bg)
            self.ax.set_facecolor(bg)

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
                        ls="--", lw=1.35, rad=1.2, z=1):
        """Draws a high-level stage container grouping related operations."""
        fc = self.get_color(palette, role="bg")
        ec = self.get_color(palette, role="border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=lw, ls=ls, rad=rad, alpha=1.0, z=z)
        if title:
            self.ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="center",
                         fontsize=10.5, fontweight="bold", color=tc, zorder=z + 2)
        return box

    def card(self, x, y, w, h, title, subtitle=None, formula=None, spec=None,
             palette="neutral", shadow=True, stack=0, rad=0.6, lw=0.95, z=3):
        """Draws an operation card with balanced typography and optional shadow/stack."""
        fc = self.get_color(palette, role="card_bg")
        ec = self.get_color(palette, role="card_border")
        tc = self.get_color(palette, role="text")
        box = self.rbox(x, y, w, h, fc=fc, ec=ec, lw=lw, rad=rad, shadow=shadow, stack=stack, z=z)

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
        self.ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                     rotation=rotation, fontsize=8.6, fontweight="bold", color=tc, zorder=z + 2)
        return box

    def arrow(self, x1, y1, x2, y2, ls="-", col="#000000", lw=1.0, hl=0.35, hw=0.22, z=4):
        """Draws a straight arrow with proportional vector arrowhead."""
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        self.ax.plot([x1, x2 - ux * hl], [y1, y2 - uy * hl],
                     linestyle=ls, color=col, lw=lw, zorder=z, solid_capstyle="round")
        self.ax.annotate("", xy=(x2, y2), xytext=(x2 - ux * hl, y2 - uy * hl),
                         arrowprops=dict(arrowstyle=f"-|>,head_length={hl*2.8},head_width={hw*2.8}",
                                         color=col, lw=lw),
                         zorder=z)

    def poly_arrow(self, pts, ls="-", col="#000000", lw=1.0, hl=0.35, hw=0.22, z=4):
        """Draws an orthogonal multi-segment path with an arrowhead on the final segment."""
        for i in range(len(pts) - 2):
            self.ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                         linestyle=ls, color=col, lw=lw, zorder=z, solid_capstyle="round")
        p_penult, p_last = pts[-2], pts[-1]
        self.arrow(p_penult[0], p_penult[1], p_last[0], p_last[1],
                   ls=ls, col=col, lw=lw, hl=hl, hw=hw, z=z)

    def plus_node(self, x, y, r=0.75, fc="#FFFFFF", ec="#000000", lw=1.0, z=6, col="#000000"):
        """Draws an element-wise addition circular node (+)."""
        c = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z)
        self.ax.add_patch(c)
        d = r * 0.55
        self.ax.plot([x - d, x + d], [y, y], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")
        self.ax.plot([x, x], [y - d, y + d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")

    def times_node(self, x, y, r=0.75, fc="#FFFFFF", ec="#000000", lw=1.0, z=6, col="#000000"):
        """Draws a matrix multiplication / Hadamard circular node (x)."""
        c = Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=z)
        self.ax.add_patch(c)
        d = r * 0.42
        self.ax.plot([x - d, x + d], [y - d, y + d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")
        self.ax.plot([x - d, x + d], [y + d, y - d], color=col, lw=lw * 1.1, zorder=z + 1, solid_capstyle="round")

    def legend(self, x, y, items, title="Legend", fc="#FFFFFF", ec="#ADB5BD", rad=0.4, z=5):
        """Draws an operation legend box with symbols and descriptions."""
        n = len(items)
        h = 2.4 + n * 1.8
        w = max(11.0, max(len(desc) for _, desc in items) * 0.65 + 4.5)
        self.rbox(x, y, w, h, fc=fc, ec=ec, lw=0.8, rad=rad, z=z)
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
        self.fig.tight_layout(pad=0.05)
        for ext in formats:
            out_file = out_dir / f"{name}.{ext}"
            tmp_file = out_dir / f"_tmp_{name}.{ext}"
            self.fig.savefig(tmp_file, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
            tmp_file.replace(out_file)
            print(f"Saved {out_file}")
        plt.close(self.fig)
