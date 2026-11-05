# Publication-Quality Matplotlib Architecture Diagram Style Guide

This style guide establishes the design language and technical standards for generating camera-ready neural network architecture diagrams, pipeline flowcharts, and technical module schematics suitable for IEEE, ACM, CVPR, NeurIPS, and Elsevier publications.

---

## 1. Canvas & Coordinate System

- **Standard Grid**: `100 × 33` units with `ax.set_aspect("equal")` and `ax.axis("off")`.
- **Figure Dimensions**: `figsize=(15.5, 5.2)` to `(16.0, 5.5)` for double-column wide figures (span full page width in 2-column LaTeX templates).
- **Coordinate Normalization**: Working in normalized `[0, 100]` units guarantees that all shapes, font proportions, and line widths remain deterministic regardless of display scaling.

---

## 2. Visual Hierarchy & Layering

1. **Stage Containers (Z-Order = 1)**:
   - Outline logical phases or pipeline stages (e.g. `1. Input`, `2. Bottleneck`, `3. Aggregation`).
   - Use soft pastel background fills (`#F8F9FA`, `#E7F5FF`, `#F3F0FF`, `#EBFBEE`, `#FFF4E6`).
   - Border style: dashed line (`ls="--"`, `lw=1.35`), corner rounding `rad=1.2` to `1.4`.
   - Title centered at top: `y = y_container + h - 1.5`, `fontsize=10.5`, `fontweight="bold"`.

2. **Drop Shadows (Z-Order = 2)**:
   - Solid offset shadow: `offset = (x + 0.22, y - 0.22)`, `fc="#000000"`, `alpha=0.08`.
   - Stacks: 2-3 offset cards behind tensors to represent batch or temporal volume (`alpha=0.6`, `offset = 0.28 * s`).

3. **Operation Cards (Z-Order = 3 to 4)**:
   - Background: Pure white `#FFFFFF` or very soft tint.
   - Border: Crisp solid border (`lw=0.9` to `1.1`), matched to the stage accent color.
   - Corner rounding: `rad=0.5` to `0.7`.

4. **Typography & Math (Z-Order = 4 to 7)**:
   - Font: Sans-serif (DejaVu Sans / Helvetica / Arial) for high readability in print.
   - Stage Titles: `10.5pt` bold.
   - Card Titles: `7.6pt` to `8.8pt` bold.
   - Equations & Tensor Shapes: `7.0pt` to `8.2pt` bold (`r"$X \in \mathbb{R}^{N \times C \times T \times V}$"`).
   - Subtitles / Comments: `6.2pt` to `6.8pt` regular or italic (`color="#555555"`).
   - Avoid long multi-line narrative text inside cards. Limit cards to **2-3 concise lines** max.

5. **Interconnection Routing (Z-Order = 4 to 6)**:
   - Straight horizontal arrows: `arrow(x1, y, x2, y)` along aligned centerlines (`cy`).
   - Symmetrical distribution bus: `poly_arrow` with shared trunk and orthogonal branches.
   - Residual shortcuts: Route along the absolute top (`y = 27.5`) or absolute bottom (`y = 3.4`), dropping cleanly into addition nodes with zero line-card intersections.

---

## 3. Mathematical Operator Nodes

- **Element-wise Addition ($\oplus$)**:
  - White filled circle (`r=0.75`, `lw=1.0`), cross `+` inside with rounded cap style.
- **Matrix Multiplication / Tensor Contraction ($\otimes$)**:
  - White filled circle (`r=0.75`, `lw=1.0`), diagonal cross `×` inside with rounded cap style.

---

## 4. Mini-Visual Components

- **Persistence Barcodes ($H_0$)**:
  - L-shaped black coordinate axes with arrowheads.
  - Horizontal blue bars of varying lengths.
  - Top bar with arrow pointing to infinity.
  - Subtitle mark `— H_0`.
- **Topological Distance Matrices (SPD)**:
  - 4×4 grid of integer geodesic hop values (`0, 1, 2, 3...`).
  - Highlighted cell in red (e.g. $d_{13} = 3$).
- **Embedding Lookup Table**:
  - Row of colored rectangle chips ($e_0, e_1, e_2, e_3, e_4$).
- **Block-Diagonal Weight Matrix**:
  - Dashed square matrix border with shaded square blocks along the main diagonal and white off-diagonal space.

---

## 5. LaTeX Integration

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{assets/example_transformer.pdf} % or .png
  \caption{Overview of the Transformer Encoder-Decoder architecture generated with Matplotlib.}
  \label{fig:transformer}
\end{figure*}
```
*(Tip: SVG files can be converted to PDF via `rsvg-convert -f pdf input.svg -o output.pdf` or `inkscape` for native vector inclusion in LaTeX).*
