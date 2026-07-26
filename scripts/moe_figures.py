"""Render the MoE attribution evidence as images for a PR or writeup.

Writes two PNGs to ``runs/proof/``:

- ``moe_coverage.png`` — a dumbbell of the share of each model's parameters
  bergson can attribute, before and after. Dumbbell because the job is
  "before -> after per item"; one hue in two shades, validated ordinal.
- ``moe_proof_terminal.png`` — the ``scripts/moe_proof.py`` run, rendered as a
  terminal window.

    python scripts/moe_figures.py
"""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

OUT_DIR = Path("runs/proof")

# Blue ramp, two ordinal steps. Validated with the dataviz palette validator:
# monotone lightness, adjacent dL >= 0.06, light end 3.54:1 on the surface,
# hue spread 2 degrees.
BEFORE = "#3987e5"
AFTER = "#0d366b"
CONNECTOR = "#b7d3f6"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

# Measured by scripts/moe_proof.py against production configs on the meta device.
COVERAGE = [
    ("gpt-oss-20b", 5.8, 97.2),
    ("Mixtral-8x7B", 3.2, 99.7),
    ("Qwen3-30B-A3B", 4.0, 99.0),
    ("OLMoE-1B-7B", 5.4, 98.5),
]


def render_coverage(path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=220)

    rows = list(enumerate(reversed(COVERAGE)))
    for y, (_, before, after) in rows:
        ax.plot(
            [before, after],
            [y, y],
            color=CONNECTOR,
            lw=2,
            zorder=1,
            solid_capstyle="round",
        )
        # 2px surface ring so the marks stay separate where they crowd.
        ax.scatter(
            [before],
            [y],
            s=110,
            color=BEFORE,
            zorder=3,
            edgecolors=SURFACE,
            linewidths=2,
        )
        ax.scatter(
            [after], [y], s=110, color=AFTER, zorder=3, edgecolors=SURFACE, linewidths=2
        )
        ax.annotate(
            f"{before:.1f}%",
            (before, y),
            textcoords="offset points",
            xytext=(-9, 0),
            ha="right",
            va="center",
            fontsize=9.5,
            color=INK_SECONDARY,
        )
        ax.annotate(
            f"{after:.1f}%",
            (after, y),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK,
            fontweight="semibold",
        )

    ax.set_yticks([y for y, _ in rows])
    ax.set_yticklabels([row[0] for _, row in rows], fontsize=10.5, color=INK)
    ax.set_xlim(-6, 112)
    ax.set_ylim(-0.7, len(COVERAGE) - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=9.5, color=INK_MUTED)

    ax.xaxis.grid(True, color=GRIDLINE, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=8)

    ax.set_title(
        "Share of parameters bergson can attribute",
        fontsize=13.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=22,
    )
    ax.text(
        0,
        1.045,
        "Fused-parameter MoE models, transformers 5.14 "
        "(measured by scripts/moe_proof.py)",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK_SECONDARY,
        va="bottom",
    )

    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=8,
                color=BEFORE,
                label="before (experts and router skipped)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                ls="",
                markersize=8,
                color=AFTER,
                label="after (experts and router tracked)",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, -0.30),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK_SECONDARY,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


TERM_BG = "#1a1a19"
TERM_INK = "#e8e7e0"
TERM_DIM = "#898781"
TERM_ACCENT = "#86b6ef"
TERM_GOOD = "#0ca30c"


def _mono(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def render_terminal(path: Path, lines: list[str]) -> None:
    """Draw ``lines`` as a terminal window."""
    scale, size = 2, 13
    font = _mono(size * scale)
    pad, chrome = 18 * scale, 34 * scale
    line_h = int(size * scale * 1.55)

    width = max(font.getbbox(ln)[2] for ln in lines) + pad * 2
    height = chrome + pad + line_h * len(lines) + pad

    img = Image.new("RGB", (width, height), TERM_BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, chrome], fill="#2c2c2a")
    for i, dot in enumerate(("#e34948", "#eda100", "#1baf7a")):
        cx = pad + i * 16 * scale
        r = 5 * scale
        draw.ellipse([cx - r, chrome // 2 - r, cx + r, chrome // 2 + r], fill=dot)
    title = "python scripts/moe_proof.py"
    draw.text(
        (width // 2 - font.getbbox(title)[2] // 2, chrome // 2 - line_h // 2),
        title,
        font=font,
        fill=TERM_DIM,
    )

    y = chrome + pad
    for line in lines:
        color = TERM_INK
        if line.startswith("=" * 10):
            color = TERM_DIM
        elif line.strip().startswith(("1.", "2.", "3.", "4.")):
            color = TERM_ACCENT
        elif "PASS" in line or line.startswith("All checks passed"):
            color = TERM_GOOD
        elif line.strip().startswith(("model ", "family ")) or "->" in line:
            color = TERM_DIM
        draw.text((pad, y), line, font=font, fill=color)
        y += line_h

    img.save(path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proof = subprocess.run(
        [sys.executable, "scripts/moe_proof.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    # Library log lines interleave with the report on stdout, and one long
    # warning would set the width of the whole image.
    noise = ("WARNING", "Warning:", "transformers]", "HF_TOKEN", "huggingface")
    lines = [
        ln.rstrip()
        for ln in proof.stdout.splitlines()
        if ln.strip() and not any(marker in ln for marker in noise)
    ]

    render_coverage(OUT_DIR / "moe_coverage.png")
    render_terminal(OUT_DIR / "moe_proof_terminal.png", lines)
    for name in ("moe_coverage.png", "moe_proof_terminal.png"):
        print(f"wrote {OUT_DIR / name}")


if __name__ == "__main__":
    main()
