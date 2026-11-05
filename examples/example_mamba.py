"""
Example 2: The Mamba Selective State Space (SSM) Block (Gu & Dao, 2023).
Demonstrates the wide / horizontal multi-branch canvas layout (FlowchartCanvas.wide).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.engine import FlowchartCanvas


def draw_mamba_block():
    # Wide horizontal layout: 100 width x 33 height
    # background="theme" -> solid theme canvas (readable on both light and dark GitHub pages)
    canvas = FlowchartCanvas.wide(figsize=(16.0, 5.5), xlim=(0, 100), ylim=(0, 33),
                                  theme="academic", background="theme")

    # STAGES
    canvas.stage_container(1.0, 1.4, 21.0, 30.2, title="1. Input & Dual Expansion", palette="neutral")
    canvas.stage_container(23.5, 1.4, 22.0, 30.2, title="2. 1D Conv & Non-linearity", palette="blue")
    canvas.stage_container(47.0, 1.4, 27.5, 30.2, title="3. Selective State Space (SSM)", palette="purple")
    canvas.stage_container(76.0, 1.4, 23.0, 30.2, title="4. Gating & Residual Output", palette="green")

    # STAGE 1: Input & Dual Projection
    cy_main = 20.0
    cy_gate = 8.0

    canvas.card(2.5, 11.5, 9.0, 8.5, title="Input Token", formula=r"$x \in \mathbb{R}^{B \times L \times D}$",
                subtitle="Sequence features", palette="neutral", stack=2, shadow=True)

    # Residual Shortcut across the top at y=28.0
    canvas.poly_arrow([(7.0, 20.0), (7.0, 28.0), (89.5, 28.0), (89.5, cy_main + 0.8)],
                      ls="--", col="#2B8A3E", lw=1.1)
    # Label box sits on the line, shifted left (x 12-48) so it clears the SSM cards (tops at y=27+)
    canvas.rbox(12.0, 27.0, 36.0, 2.0, fc="#FFFFFF", ec="#2B8A3E", lw=0.75, rad=0.4, z=5)
    canvas.ax.text(30.0, 28.0, r"Block Residual Shortcut: $x + \mathrm{Linear}(\mathrm{SSM}(x) \odot \mathrm{Gate}(x))$",
                   ha="center", va="center", fontsize=7.2, fontweight="bold", color="#2B8A3E", zorder=6)

    # Branching from Input
    canvas.poly_arrow([(11.5, 15.75), (13.5, 15.75), (13.5, cy_main), (15.0, cy_main)], lw=1.0)
    canvas.poly_arrow([(11.5, 15.75), (13.5, 15.75), (13.5, cy_gate), (15.0, cy_gate)], lw=1.0)

    # Branch 1: Main Path Linear Projection
    canvas.card(15.0, cy_main - 3.0, 6.0, 6.0, title="Linear", formula=r"$D \to E$",
                spec=r"$(E = 2D)$", palette="blue", shadow=True)

    # Branch 2: Gating Path Linear Projection
    canvas.card(15.0, cy_gate - 3.0, 6.0, 6.0, title="Linear", formula=r"$D \to E$",
                spec="Gate path", palette="orange", shadow=True)

    # STAGE 2: 1D Conv & SiLU
    canvas.arrow(21.0, cy_main, 25.0, cy_main, lw=1.0)
    canvas.card(25.0, cy_main - 3.2, 9.0, 6.4, title="1D Conv", formula=r"$\mathrm{Conv1d}(k=4)$",
                spec="Depthwise temporal", palette="blue", shadow=True)

    canvas.arrow(34.0, cy_main, 36.5, cy_main, lw=1.0)
    canvas.card(36.5, cy_main - 3.2, 7.5, 6.4, title="SiLU", formula=r"$x \cdot \sigma(x)$",
                spec="Non-linearity", palette="blue", shadow=True)

    # Gate path passes through SiLU
    canvas.arrow(21.0, cy_gate, 36.5, cy_gate, lw=1.0)
    canvas.card(36.5, cy_gate - 3.2, 7.5, 6.4, title="SiLU", formula=r"$z \cdot \sigma(z)$",
                spec="Gate activation", palette="orange", shadow=True)

    # STAGE 3: Selective SSM (Core Innovation of Mamba)
    canvas.arrow(44.0, cy_main, 48.5, cy_main, lw=1.0)

    # Parameter discretization & selectivity
    canvas.card(48.5, 18.0, 11.5, 9.0, title="Parameter Discretization",
                formula=r"$\Delta = \mathrm{softplus}(\mathrm{Lin}(u) + \Delta_b)$",
                spec=r"$\bar{A} = \exp(\Delta A),\ \bar{B} = (\Delta A)^{-1}(\bar{A}-I)\Delta B$",
                palette="purple", shadow=True)

    # Selective scan module
    canvas.arrow(60.0, 22.5, 63.0, 22.5, lw=1.0)
    canvas.card(63.0, 17.5, 10.0, 10.0, title="Selective Scan",
                formula=r"$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$",
                subtitle=r"$y_t = C_t h_t$",
                spec=r"$O(L)\ \mathrm{linear\ time!}$", palette="purple", shadow=True)

    # STAGE 4: Multiplicative Gating & Residual Output
    # Gate arrow routes from bottom to Hadamard node
    canvas.arrow(44.0, cy_gate, 76.5, cy_gate, lw=1.0)

    # Hadamard Multiplication (x) Node
    canvas.poly_arrow([(73.0, 22.5), (78.5, 22.5), (78.5, 15.5)], col="#7048E8", lw=1.0)
    canvas.poly_arrow([(76.5, cy_gate), (78.5, cy_gate), (78.5, 13.9)], col="#D9480F", lw=1.0)

    cx_gate_mul = 78.5
    cy_gate_mul = 14.7
    canvas.times_node(cx_gate_mul, cy_gate_mul, r=0.75, col="#D9480F")
    # Placed right of the incoming vertical arrow (x=78.5) and above the Linear card (top y=17.9)
    canvas.ax.text(cx_gate_mul + 0.7, 18.7, r"$y \odot \mathrm{SiLU}(z)$", ha="left", va="center",
                   fontsize=7.2, fontweight="bold", color="#D9480F", zorder=7)

    # Linear projection back to dimension D
    canvas.arrow(cx_gate_mul + 0.8, cy_gate_mul, 82.0, cy_gate_mul, lw=1.0)
    canvas.card(82.0, cy_gate_mul - 3.2, 5.5, 6.4, title="Linear", formula=r"$E \to D$", palette="green", shadow=True)

    # Residual addition node (+)
    canvas.arrow(87.5, cy_gate_mul, 89.5 - 0.8, cy_gate_mul, lw=1.0)
    # Residual line descends from y=28 to cy_gate_mul
    canvas.poly_arrow([(89.5, 28.0), (89.5, cy_gate_mul + 0.8)], col="#2B8A3E", lw=1.1)
    canvas.plus_node(89.5, cy_gate_mul, r=0.75, col="#2B8A3E")

    # Final Mamba Output Card
    canvas.arrow(89.5 + 0.8, cy_gate_mul, 92.5, cy_gate_mul, lw=1.0)
    canvas.card(92.5, cy_gate_mul - 4.5, 5.8, 9.0, title="Mamba\nOutput", formula=r"$y \in \mathbb{R}^{B \times L \times D}$",
                palette="green", stack=2, shadow=True)

    # Legend
    # Free pocket in stage 3: above the gate arrow (y=8), below the SSM cards (bottoms y>=17.5)
    canvas.legend(57.5, 9.0, [
        ("+", "Element-wise Sum"),
        ("x", "Hadamard Gating"),
        ("--", "Residual Shortcut"),
    ], title="Notation")

    # Output save
    out_dir = Path(__file__).parent.parent / "assets"
    canvas.save(out_dir, "example_mamba")


if __name__ == "__main__":
    draw_mamba_block()
