#!/usr/bin/env python3
"""Build an N-row comparison gallery of surround-view synthesis.

Rows are given on the CLI as `--row LABEL:KIND:ROOT` triples, top-to-bottom:
  KIND=pred  -> reads ROOT/**/clip_*/pred_<view>.mp4
  KIND=gt    -> reads ROOT/**/clip_*/gt_<view>.mp4  (or ROOT/<uuid>/gt_<view>.mp4)
Every row is matched to the others by the 8-char uuid prefix.

Example — the geometry-vs-caption 4-way:
  python tools/make_nway_viz.py --out demo/outputs/gallery_ctrl \
    --row "RAW (own Qwen caption):pred:demo/outputs/raw_e2e/renders" \
    --row "RAW-ctrl (cached caption):pred:demo/outputs/raw_e2e/renders_ctrl" \
    --row "CACHED:pred:demo/outputs/cached_renders" \
    --row "REAL GT:gt:demo/gt"
"""
from __future__ import annotations
import argparse, glob, os, re
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

COL_VIEWS = ["cross_left", "rear_left", "rear_tele", "rear_right", "cross_right"]
NICE = {"cross_left": "CROSS-L (side)", "rear_left": "REAR-L", "rear_tele": "REAR-TELE",
        "rear_right": "REAR-R", "cross_right": "CROSS-R (side)"}
TW, TH = 288, 168
GAP, LABEL_W, COLHDR = 8, 150, 26
BANNER_SCALE = 1.7
FPS, N_FRAMES = 10, 41
BG, FG = (14, 14, 16), (238, 238, 238)
# a distinct colour per row slot (amber, teal, green, cyan, magenta, ...)
PALETTE = [(255, 196, 60), (120, 210, 200), (120, 230, 140), (90, 200, 255),
           (230, 140, 230), (240, 240, 240)]


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


F_LG, F_MD, F_SM = _font(19), _font(15), _font(12)


def load_vid(path, n=N_FRAMES):
    if path is None or not Path(path).exists():
        return None
    try: return iio.imread(path, plugin="pyav")[:n]
    except Exception: return None


def fit(frame, tw, th):
    im = Image.fromarray(frame).convert("RGB"); im.thumbnail((tw, th), Image.BILINEAR)
    c = Image.new("RGB", (tw, th), (0, 0, 0)); c.paste(im, ((tw-im.width)//2, (th-im.height)//2))
    return c


def missing(tw, th, msg="— no data —"):
    c = Image.new("RGB", (tw, th), (40, 30, 30))
    ImageDraw.Draw(c).text((tw//2-40, th//2-8), msg, font=F_SM, fill=(200, 160, 160))
    return c


def _prefix(name: str) -> str:
    m = re.search(r"[0-9a-f]{8}", name)
    return m.group(0) if m else name[:8]


def index_source(root: Path, kind: str):
    """{prefix: clip_dir} for dirs under root holding the row's videos."""
    out = {}
    pat = "pred_cross_left.mp4" if kind == "pred" else "gt_cross_left.mp4"
    for p in glob.glob(str(root / "**" / pat), recursive=True):
        cd = Path(p).parent
        name = cd.parent.name if cd.name.startswith("clip_") else cd.name
        out.setdefault(_prefix(name), cd)
    # gt fallback: <root>/<uuid>/gt_*.mp4 (no clip_ level)
    if kind == "gt" and not out:
        for p in glob.glob(str(root / "*" / pat)):
            cd = Path(p).parent
            out.setdefault(_prefix(cd.name), cd)
    return out


def compose(frames, fi, name, rows):
    ncol = len(COL_VIEWS)
    grid_w = LABEL_W + ncol*TW + (ncol-1)*GAP
    bw, bh = int(TW*BANNER_SCALE), int(TH*BANNER_SCALE)
    y_col = bh + 34
    row_h = TH + GAP + 20
    y_rows = [y_col + COLHDR + r*row_h for r in range(len(rows))]
    H = y_rows[-1] + TH + 14
    W = max(grid_w, bw + 420) + 2*GAP
    W += W & 1; H += H & 1
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)

    def fr(key):
        a = frames.get(key)
        return None if a is None or len(a) == 0 else a[min(fi, len(a)-1)]

    f0 = fr("front"); bx = GAP
    canvas.paste(fit(f0, bw, bh) if f0 is not None else missing(bw, bh), (bx, 30))
    d.rectangle([bx, 30, bx+bw, 30+bh], outline=(90, 90, 90), width=1)
    d.text((bx, 6), "FRONT CAMERA  —  the ONLY real input the world-model sees", font=F_MD, fill=FG)

    tx = bx + bw + 18
    d.text((tx, 34), f"clip  {name[:22]}", font=F_MD, fill=FG)
    d.text((tx, 58), f"frame {fi+1:02d}/{N_FRAMES}", font=F_SM, fill=(170, 170, 170))
    yy = 84
    for (label, _, _), col in zip(rows, PALETTE):
        d.text((tx, yy), label, font=F_SM, fill=col); yy += 18

    for c, v in enumerate(COL_VIEWS):
        x = LABEL_W + c*(TW+GAP)
        d.text((x, y_col+4), NICE[v], font=F_SM, fill=FG)

    for r, ((label, _, _), y) in enumerate(zip(rows, y_rows)):
        col = PALETTE[r % len(PALETTE)]
        # wrap the row label into the LABEL_W gutter
        words = label.split(); line = ""; ly = y + 6
        for w in words:
            probe = (line + " " + w).strip()
            if len(probe) > 15 and line:
                d.text((GAP, ly), line, font=F_SM, fill=col); ly += 15; line = w
            else:
                line = probe
        if line: d.text((GAP, ly), line, font=F_SM, fill=col)
        for c, v in enumerate(COL_VIEWS):
            x = LABEL_W + c*(TW+GAP)
            f = fr(f"r{r}_{v}")
            canvas.paste(fit(f, TW, TH) if f is not None else missing(TW, TH), (x, y))
            d.rectangle([x, y, x+TW, y+TH], outline=col, width=2)
    return np.asarray(canvas)


def write_mp4(path, frames):
    frames = np.stack(frames)
    for codec in ("libx264", "mpeg4"):
        try: iio.imwrite(path, frames, plugin="pyav", codec=codec, fps=FPS); return codec
        except Exception: continue
    iio.imwrite(str(path).replace(".mp4", ".gif"), frames, plugin="pillow", duration=1000//FPS, loop=0)
    return "gif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True,
                    help="LABEL:KIND:ROOT  (KIND=pred|gt). Repeat, top-to-bottom.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="OpenLongTail — surround synthesis comparison")
    ap.add_argument("--sub", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.row:
        label, kind, root = spec.split(":", 2)
        assert kind in ("pred", "gt"), f"bad KIND in {spec!r}"
        rows.append((label, kind, Path(root)))
    idx = [index_source(root, kind) for (_, kind, root) in rows]
    keys = sorted(set().union(*[set(i) for i in idx]))
    if not keys:
        raise SystemExit("no clips found in any row")
    for (label, _, _), i in zip(rows, idx):
        print(f"row {label!r}: {len(i)} clips")
    print(f"-> {len(keys)} clips union")

    cards = []
    for k, pref in enumerate(keys, 1):
        frames = {"front": None}
        for r, ((label, kind, _), i) in enumerate(zip(rows, idx)):
            cd = i.get(pref)
            if frames["front"] is None and cd is not None:
                frames["front"] = load_vid(cd / "front_input.mp4")
            for v in COL_VIEWS:
                stem = "pred" if kind == "pred" else "gt"
                frames[f"r{r}_{v}"] = load_vid(cd / f"{stem}_{v}.mp4") if cd else None
        T = min(max([len(a) for a in frames.values() if a is not None] + [1]), N_FRAMES)
        seq = [compose(frames, fi, pref, rows) for fi in range(T)]
        Image.fromarray(seq[T//2]).save(args.out / f"{k:02d}_{pref}.png")
        codec = write_mp4(str(args.out / f"{k:02d}_{pref}.mp4"), seq)
        cards.append((f"{k:02d}_{pref}", pref))
        have = "".join(str(r) for r, i in enumerate(idx) if pref in i)
        print(f"  [{k}/{len(keys)}] {pref}  rows=[{have}]  +mp4({codec})")

    sub = args.sub or ("Each clip: the world-model sees <b>only the front camera</b> (top banner) and "
                       "generates the 5 side/rear rig views. Rows compared, aligned by column.")
    card_html = "\n".join(
        f'<div class="card"><div class="lbl">{n}</div>'
        f'<video src="{fn}.mp4" controls loop muted playsinline preload="none" poster="{fn}.png"></video></div>'
        for fn, n in cards)
    (args.out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>{args.title}</title>
<style>body{{margin:0;background:#0e0e10;color:#eee;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:24px 28px 4px;font-size:22px}}.sub{{color:#9a9a9a;margin:0 28px 16px;max-width:1150px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:16px;padding:8px 28px 32px}}
.card{{background:#191920;border:1px solid #24242e;border-radius:10px;overflow:hidden}}
.card video{{width:100%;display:block;background:#000}}.lbl{{padding:8px 12px;color:#ffc43c;font-family:monospace;font-size:13px}}</style>
<h1>{args.title}</h1><div class="sub">{sub}</div><div class="grid">{card_html}</div>""")
    print(f"\nwrote {args.out/'index.html'}  ({len(cards)} clips)")


if __name__ == "__main__":
    main()
