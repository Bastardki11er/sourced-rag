"""Renders docs/demo/sourced-demo.gif, a terminal output replay, from docs/demo/demo-input.json.

    python docs/demo/render_demo.py [--font /path/to/monospace.ttf]

Needs only Pillow (already installed with the runtime requirements) and a monospace font. It reads the
input file next to this script, makes no network or API calls, and writes the GIF next to it. Frames show
the captured text unchanged; waits are not reproduced, each step is simply held long enough to read.
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONTS = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
W, H, PAD, LINE, COLS = 1040, 640, 22, 20, 100
BG, TEXT, DIM, CMD, BAR, BAR_TEXT, HISTORY = "#1e1f22", "#e6e6e6", "#9aa0a6", "#8fd18f", "#2f5d8a", "#fff", "#7a4a00"
REPLAY = "Terminal output replay of a real run (not a screen recording). Extractive mode: no Claude call."
STEPS = [  # (title, seconds the output stays on screen)
    ("1/4  Index the sample handbook (new temporary index)", 4),
    ("2/4  Retrieval only: ranked passages, with where BM25 and dense search ranked them", 13),
    ("3/4  Ask: extractive answer = the top passage, with a numbered citation", 15),
    ("4/4  Open the cited source: the quote is the 'Rolling back' section, word for word", 11),
]


def wrap(text: str) -> list[str]:
    return [part for line in text.splitlines() for part in textwrap.wrap(line, COLS, subsequent_indent="    ") or [""]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--font", help="a monospace .ttf/.ttc file")
    args = parser.parse_args()
    path = args.font or next((f for f in FONTS if Path(f).exists()), None)
    if path is None:
        raise SystemExit("no monospace font found; pass one with --font")
    font, bold = ImageFont.truetype(path, 15), ImageFont.truetype(path, 15, index=1 if path.endswith(".ttc") else 0)

    def frame(title, lines, bar=BAR, footer=REPLAY):
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, W, 34], fill=bar)
        draw.text((PAD, 8), title, font=bold, fill=BAR_TEXT)
        draw.rectangle([0, H - 26, W, H], fill="#2a2b2e")
        draw.text((PAD, H - 21), footer, font=font, fill=DIM)
        for i, (style, line) in enumerate(lines):
            color = {"cmd": CMD, "dim": DIM}.get(style, TEXT)
            draw.text((PAD, 48 + i * LINE), line, font=bold if style == "cmd" else font, fill=color)
        return img

    data = json.loads((HERE / "demo-input.json").read_text())
    run, past = data["extractive_run"], data["historical_claude"]
    frames = [
        (
            frame(
                "Sourced demo: ask a question, see the passage, check the source",
                [
                    ("", ""),
                    ("", "Real output from a local run on 2026-10-29, sample handbook (fictional company)."),
                    ("", "Answerer: extractive. It returns the best retrieved passage verbatim; no language model,"),
                    ("", "no Claude call. Waits (model loading) shortened; output text unchanged."),
                    ("", ""),
                    ("dim", "Setup: " + run["setup"]),
                    ("", ""),
                    ("", "Note: `kctl` is a command inside the fictional handbook. Don't run it."),
                ],
            ),
            7,
        )
    ]
    for step, (title, hold) in zip(run["steps"], STEPS, strict=True):
        command = [("cmd", "$ " + step["cmd"])]
        frames.append((frame(title, command), 1.5))
        output = [("", line) for line in wrap(step["stdout"].rstrip())]
        frames.append((frame(title, command + output + [("dim", f"[exit {step['exit']}]")]), hold))
    cite = past["citation"]
    frames.append(
        (
            frame(
                "HISTORICAL CALL RECORD (not generated in this demo)",
                [
                    (
                        "dim",
                        f"Recorded 2026-10-29 UTC by the live check: {past['model']}, effort {past['effort']}, "
                        f"stop_reason {past['stop_reason']}.",
                    ),
                    ("dim", "Question: " + past["question"]),
                    ("", ""),
                    *[("", line) for line in wrap(past["answer"])],
                    ("", ""),
                    ("dim", f"[{cite['number']}] {cite['source']} > {cite['heading']}"),
                    ("dim", "    quote = the whole 'Rolling back' section shown in step 4"),
                    ("", ""),
                    ("dim", "One example from a 12-question check; see docs/evaluation.md for all results and limits."),
                ],
                bar=HISTORY,
                footer="Text of a stored record, shown as is. This demo made no Claude call.",
            ),
            9,
        )
    )
    images, seconds = zip(*frames, strict=True)
    out = HERE / "sourced-demo.gif"
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=[int(s * 1000) for s in seconds], loop=0, optimize=True
    )
    print(f"wrote {out.relative_to(HERE.parent.parent)}: {len(images)} frames, {sum(seconds):.0f} s")


if __name__ == "__main__":
    main()
