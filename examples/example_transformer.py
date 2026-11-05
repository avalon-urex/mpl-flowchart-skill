"""
Example 1: The Iconic Transformer Architecture ("Attention Is All You Need", Vaswani et al., 2017).
Demonstrates the vertical / tall canvas layout (FlowchartCanvas.tall).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.engine import FlowchartCanvas


def draw_transformer():
    # Vertical canvas: 65 width x 100 height
    # background="theme" -> solid theme canvas (readable on both light and dark GitHub pages)
    canvas = FlowchartCanvas.tall(figsize=(9.5, 12.5), xlim=(0, 65), ylim=(0, 100),
                                  theme="academic", background="theme")

    # Title
    canvas.ax.text(32.5, 97.5, "The Transformer Architecture (Vaswani et al., 2017)",
                   ha="center", va="center", fontsize=11.5, fontweight="bold", color="#212529")

    # ENCODER STACK (Left Column, x in [4, 30])
    canvas.stage_container(4.0, 24.0, 26.0, 56.0, title="Encoder (xN)", palette="neutral", rad=1.4)

    # Input Embedding & Positional Encoding
    canvas.card(8.0, 3.0, 18.0, 4.5, title="Inputs", palette="neutral", shadow=True)
    canvas.arrow(17.0, 7.5, 17.0, 11.0, lw=1.0)
    canvas.card(8.0, 11.0, 18.0, 5.0, title="Input Embedding", formula=r"$X \in \mathbb{R}^{B \times L \times d_{\mathrm{model}}}$", palette="blue", shadow=True)
    canvas.arrow(17.0, 16.0, 17.0, 19.5, lw=1.0)

    # Positional Encoding (+)
    canvas.plus_node(17.0, 20.2, r=0.75, col="#1C7ED6")
    canvas.card(1.0, 18.2, 12.0, 4.0, title="Positional\nEncoding", palette="yellow", rad=0.5, shadow=True)
    canvas.arrow(13.0, 20.2, 16.2, 20.2, lw=0.9)
    canvas.arrow(17.0, 21.0, 17.0, 26.0, lw=1.0)

    # Encoder Sub-layer 1: Multi-Head Attention
    canvas.card(7.0, 26.0, 20.0, 6.5, title="Multi-Head Attention", formula=r"$\mathrm{Softmax}(QK^\top / \sqrt{d_k})V$",
                spec=r"$h = 8\ \mathrm{heads},\ d_k = 64$", palette="orange", shadow=True)
    canvas.arrow(17.0, 32.5, 17.0, 36.5, lw=1.0)

    # Add & Norm 1
    canvas.card(7.0, 36.5, 20.0, 5.0, title="Add & Norm", formula=r"$\mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$", palette="yellow", shadow=True)

    # Skip connection around Multi-Head Attention
    canvas.poly_arrow([(17.0, 23.5), (4.8, 23.5), (4.8, 39.0), (7.0, 39.0)],
                      ls="--", col="#D9480F", lw=1.0)

    canvas.arrow(17.0, 41.5, 17.0, 46.0, lw=1.0)

    # Encoder Sub-layer 2: Feed Forward
    canvas.card(7.0, 46.0, 20.0, 6.5, title="Feed Forward", formula=r"$\mathrm{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$",
                spec=r"$d_{\mathrm{ff}} = 2048$", palette="blue", shadow=True)
    canvas.arrow(17.0, 52.5, 17.0, 56.5, lw=1.0)

    # Add & Norm 2
    canvas.card(7.0, 56.5, 20.0, 5.0, title="Add & Norm", formula=r"$\mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$", palette="yellow", shadow=True)

    # Skip connection around Feed Forward
    canvas.poly_arrow([(17.0, 43.5), (4.8, 43.5), (4.8, 59.0), (7.0, 59.0)],
                      ls="--", col="#1C7ED6", lw=1.0)

    # DECODER STACK (Right Column, x in [35, 61])
    canvas.stage_container(35.0, 24.0, 26.0, 56.0, title="Decoder (xN)", palette="neutral", rad=1.4)

    # Output Embedding & Positional Encoding
    canvas.card(39.0, 3.0, 18.0, 4.5, title="Outputs (shifted right)", palette="neutral", shadow=True)
    canvas.arrow(48.0, 7.5, 48.0, 11.0, lw=1.0)
    canvas.card(39.0, 11.0, 18.0, 5.0, title="Output Embedding", formula=r"$Y \in \mathbb{R}^{B \times L \times d_{\mathrm{model}}}$", palette="blue", shadow=True)
    canvas.arrow(48.0, 16.0, 48.0, 19.5, lw=1.0)

    # Positional Encoding (+)
    canvas.plus_node(48.0, 20.2, r=0.75, col="#1C7ED6")
    canvas.card(53.0, 18.2, 11.0, 4.0, title="Positional\nEncoding", palette="yellow", rad=0.5, shadow=True)
    canvas.arrow(53.0, 20.2, 48.8, 20.2, lw=0.9)
    canvas.arrow(48.0, 21.0, 48.0, 26.0, lw=1.0)

    # Decoder Sub-layer 1: Masked Multi-Head Attention
    canvas.card(38.0, 26.0, 20.0, 6.5, title="Masked Multi-Head Attn", formula=r"$\mathrm{Softmax}(QK^\top / \sqrt{d_k} + M)V$",
                spec="Causal future masking", palette="orange", shadow=True)
    canvas.arrow(48.0, 32.5, 48.0, 36.5, lw=1.0)

    # Add & Norm 1 (Decoder)
    canvas.card(38.0, 36.5, 20.0, 5.0, title="Add & Norm", formula=r"$\mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$", palette="yellow", shadow=True)
    canvas.poly_arrow([(48.0, 23.5), (60.2, 23.5), (60.2, 39.0), (58.0, 39.0)],
                      ls="--", col="#D9480F", lw=1.0)

    canvas.arrow(48.0, 41.5, 48.0, 46.0, lw=1.0)

    # Cross-Attention (Multi-Head Attention from Encoder to Decoder)
    canvas.card(38.0, 46.0, 20.0, 6.5, title="Multi-Head Attention", formula=r"Query $\leftarrow$ Dec, Key/Value $\leftarrow$ Enc",
                spec="Cross-modal conditioning", palette="orange", shadow=True)

    # Arrow from Encoder top to Decoder Cross-Attention
    canvas.poly_arrow([(17.0, 61.5), (17.0, 68.0), (33.0, 68.0), (33.0, 49.2), (38.0, 49.2)],
                      col="#7048E8", lw=1.1)

    canvas.arrow(48.0, 52.5, 48.0, 56.5, lw=1.0)

    # Add & Norm 2 (Decoder)
    canvas.card(38.0, 56.5, 20.0, 5.0, title="Add & Norm", formula=r"$\mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$", palette="yellow", shadow=True)
    canvas.poly_arrow([(48.0, 43.5), (60.2, 43.5), (60.2, 59.0), (58.0, 59.0)],
                      ls="--", col="#7048E8", lw=1.0)

    canvas.arrow(48.0, 61.5, 48.0, 66.0, lw=1.0)

    # Decoder Sub-layer 3: Feed Forward
    canvas.card(38.0, 66.0, 20.0, 6.0, title="Feed Forward", formula=r"$\mathrm{FFN}(x) = \mathrm{ReLU}(xW_1 + b_1)W_2 + b_2$",
                palette="blue", shadow=True)
    canvas.arrow(48.0, 72.0, 48.0, 74.5, lw=1.0)

    # Add & Norm 3 (Decoder)
    canvas.card(38.0, 74.5, 20.0, 4.5, title="Add & Norm", formula=r"$\mathrm{LayerNorm}(x + \mathrm{SubLayer}(x))$", palette="yellow", shadow=True)
    canvas.poly_arrow([(48.0, 63.5), (60.2, 63.5), (60.2, 76.5), (58.0, 76.5)],
                      ls="--", col="#1C7ED6", lw=1.0)

    # FINAL PROJECTION HEAD
    canvas.arrow(48.0, 79.0, 48.0, 83.0, lw=1.0)
    canvas.card(38.0, 83.0, 20.0, 4.5, title="Linear Projection", formula=r"$d_{\mathrm{model}} \to d_{\mathrm{vocab}}$", palette="purple", shadow=True)
    canvas.arrow(48.0, 87.5, 48.0, 90.0, lw=1.0)
    canvas.card(38.0, 90.0, 20.0, 4.5, title="Softmax", formula=r"$P(y_t | y_{<t}, X) = \mathrm{Softmax}(z)$", palette="green", shadow=True)

    # Legend
    canvas.legend(4.0, 83.0, [
        ("+", "Add (Residual)"),
        ("--", "Skip Connection"),
        ("x", "Cross Attention")
    ], title="Notation")

    # Output save
    out_dir = Path(__file__).parent.parent / "assets"
    canvas.save(out_dir, "example_transformer")


if __name__ == "__main__":
    draw_transformer()
