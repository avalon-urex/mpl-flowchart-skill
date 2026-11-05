---
name: matplotlib-flowchart
description: >-
  Use this skill to design, generate, and customize publication-quality deep learning
  architecture diagrams, neural network flowcharts, and technical pipeline figures
  in Python using Matplotlib with arbitrary canvas sizes, flexible themes, and vector SVG / PNG exports.
---

# Universal Matplotlib Architecture & Flowchart Skill

A complete, production-ready framework for programmatically generating publication-quality deep learning architecture diagrams, module flowcharts, and spatio-temporal pipeline schematics using Python and Matplotlib.

---

## When to Use This Skill

Activate this skill when:
- Designing new neural network architecture diagrams (e.g., Transformers, Mamba SSM, ConvNets, Diffusion pipelines).
- Creating module deep-dives (e.g., multi-head attention, selective state spaces, bottleneck blocks, residual convolutions).
- Needing arbitrary canvas aspect ratios:
  - **Wide horizontal pipelines** (e.g., multi-stage pipelines, multi-scale conv blocks).
  - **Tall vertical stacks** (e.g., Transformer Encoder-Decoder, deep sequential layers).
  - **Square / single-module zooms** (e.g., attention gate, spatial message passing).
- Customizing color themes (Academic pastel, Modern Dark mode, Minimal Grayscale for B&W print, or custom palettes).
- Exporting publication-grade vector graphics (`.svg`) and high-DPI raster images (`.png`) for LaTeX, Overleaf, Word, or presentations.

---

## Canvas Layout Modes & Sizing

The engine supports arbitrary canvas sizes via `FlowchartCanvas`:

| Layout Mode | Constructor | Default Dimensions | Best For |
| :--- | :--- | :--- | :--- |
| **Wide Banner** | `FlowchartCanvas.wide()` | `(16.0, 5.5)`, `xlim=(0, 100)`, `ylim=(0, 33)` | Horizontal pipelines, multi-stage modules |
| **Tall Stack** | `FlowchartCanvas.tall()` | `(9.5, 12.5)`, `xlim=(0, 65)`, `ylim=(0, 100)` | Encoder-Decoder stacks, deep sequential layers |
| **Square** | `FlowchartCanvas.square()` | `(8.0, 8.0)`, `xlim=(0, 50)`, `ylim=(0, 50)` | Single building blocks, attention matrices |
| **Custom** | `FlowchartCanvas(figsize, xlim, ylim)` | User-defined | Any arbitrary diagram requirements |

---

## Theming & Color Customization

Choose from built-in palettes or pass custom hex values:
- `"academic"` / `"academic_light"`: Curated soft pastels with crisp high-contrast text (IEEE, CVPR, NeurIPS default).
- `"modern_dark"`: Dark canvas with glowing accents (ideal for GitHub READMEs and slides).
- `"minimal_mono"`: Clean black-and-white print with high-contrast borders.

Example:
```python
canvas = FlowchartCanvas.wide(theme="academic")  # or "dark", "mono"
# background="theme" (default: solid theme canvas) or background="trans" (transparent)
# Direct hex override is always supported anywhere:
canvas.rbox(x, y, w, h, fc="#E7F5FF", ec="#1C7ED6")
```

---

## Core Primitives Reference

- **Stage Container**:
  `canvas.stage_container(x, y, w, h, title="1. Stage Name", palette="blue")`
- **Operation Card**:
  `canvas.card(x, y, w, h, title="Linear", formula=r"$D \to E$", subtitle="Gate path", palette="orange", shadow=True, stack=2)`
- **Feature Pill**:
  `canvas.pill(x, y, w, h, label=r"Token CLS $c_i$", badge="1,280", palette="blue")`
- **Vertical Spine Block**:
  `canvas.vertical_block(x, y, w, h, label="Backbone ViT-H", rotation=90, palette="purple")`
- **Orthogonal Arrow**:
  `canvas.poly_arrow([(x1, y1), (x2, y2), (x3, y3)], col="#1C7ED6", lw=1.0)`
- **Math Nodes**:
  `canvas.plus_node(x, y, r=0.75)` (Element-wise Sum)
  `canvas.times_node(x, y, r=0.75)` (Hadamard / Matrix Multiplication)
- **Notation Legend**:
  `canvas.legend(x, y, [("+", "Element-wise Sum"), ("x", "Gating"), ("--", "Skip Connection")])`
- **Camera-Ready Export**:
  `canvas.save("output_dir", "figure_name")` (Outputs both `.png` and `.svg`)

---

## Layout Workflow: Render, Inspect, Fix, Repeat

The engine does **not** auto-place or auto-avoid anything: every coordinate is yours. Treat the first render as a draft and iterate until nothing overlaps. Never hand over a figure you have not looked at.

**Loop (usually 2-5 rounds):**
1. Run the script to produce the PNG (`python my_diagram.py`).
2. **Look at the image** (open the PNG with the image-capable Read tool). For dense regions, crop/zoom a region (e.g. with PIL) so small overlaps are visible. Don't judge from the code alone.
3. Check the list below and note each problem with its approximate data coordinates.
4. Edit the coordinates in the script (move, resize, re-route), then re-render and look again.
5. Stop only when a full pass finds no problems.

**Overlap checklist:**
- Label/annotation text vs. cards, nodes, arrows, or other text.
- Boxed labels on a path vs. the cards or titles next to the path.
- Legend vs. arrows, cards, or stage borders (it should sit entirely in a free pocket).
- Stage titles vs. arrows entering/leaving the container (title hidden behind a card or crossed by a line).
- Containers too short for their contents (children poking past the border, title colliding with the top card).
- Arrowheads landing inside or on top of nodes/cards; arrows crossing text.
- Anything cut off at the canvas edge, or too close (< ~1 unit) to a neighbor.

**Placement rules of thumb (data units):**
- Reserve space first: decide each stage/column extent, then position cards inside with at least ~1.5 units of margin; give containers extra height for the title row (~3 units).
- Estimate text size before placing it: about 0.55-0.6 units per character at font size 7-8 on the default wide canvas (100 units across 16 in); scale with `xlim` and `figsize`. Make boxes 10-20% wider than the estimate.
- Slide a path label along its line (the box can sit anywhere on the line) until it clears neighboring cards; shift it into a stage with empty vertical space.
- Put a node's caption in free space beside it, not on the incoming/outgoing arrows; a side with a card or vertical arrow is not free.
- Put the legend / notation key in a canvas corner (lower-right or upper-left is usual), clear of arrows and cards. If every corner is occupied, shrink or move the neighboring card (e.g. make a card shorter around its arrow) before falling back to an empty pocket inside the diagram. Check both the box and its title row.
- Keep arrows straight and orthogonal; if two cross, reroute one around with `poly_arrow` waypoints rather than shrinking cards.
- Move title text off-center (`canvas.ax.text(..., ha="left")` with `title=None` on the container) if an arrow enters the container at its middle.
- Change one thing at a time when fixing and re-render, so fixes don't create new overlaps.

---

## Landmark Reference Examples in Repository

1. **Transformer Architecture (`examples/example_transformer.py`)**:
   Full Encoder-Decoder stack from *"Attention Is All You Need"* (Vaswani et al., 2017) showcasing vertical multi-column layout, positional encodings, cross-attention routing, and skip connections.
2. **Mamba Selective SSM Block (`examples/example_mamba.py`)**:
   Selective State Space block from *"Mamba"* (Gu & Dao, 2023) showcasing horizontal dual-branch expansion, discretization, selective scan, Hadamard gating, and residual bypass.

To generate all reference diagrams:
```bash
python examples/example_transformer.py
python examples/example_mamba.py
```
