#!/usr/bin/env python3
"""Build the OpenLongTail demo gallery.

Given a renders dir (from inference / inference_cached) with per-clip
`front_input.mp4` + `pred_<view>.mp4`, produce for each clip a temporal-mosaic
comparison video and a still, then a self-contained index.html.

Layout per frame: front-input banner on top (the ONLY real input), then a
PRED row (amber). With --with-gt, a REAL-GT row (cyan) is added below,
column-aligned, from a parallel gt dir (front_input.mp4 + gt_<view>.mp4).

  python tools/make_demo_viz.py --renders <dir> --out <dir>
  python tools/make_demo_viz.py --renders <dir> --gt <dir> --out <dir> --with-gt
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont

# spatial left->right around the car
COL_VIEWS = ["cross_left", "rear_left", "rear_tele", "rear_right", "cross_right"]
NICE = {"cross_left": "CROSS-L (side)", "rear_left": "REAR-L", "rear_tele": "REAR-TELE",
        "rear_right": "REAR-R", "cross_right": "CROSS-R (side)"}
TW, TH = 288, 168
GAP, LABEL_W, COLHDR = 8, 118, 26
BANNER_SCALE = 1.7
FPS, N_FRAMES = 10, 41
BG, FG = (14, 14, 16), (238, 238, 238)
PRED_C, GT_C = (255, 196, 60), (90, 200, 255)


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


F_LG, F_MD, F_SM = _font(20), _font(15), _font(13)


def load_vid(path, n=N_FRAMES):
    try: return iio.imread(path, plugin="pyav")[:n]
    except Exception: return None


def fit(frame, tw, th):
    im = Image.fromarray(frame).convert("RGB"); im.thumbnail((tw, th), Image.BILINEAR)
    c = Image.new("RGB", (tw, th), (0, 0, 0)); c.paste(im, ((tw-im.width)//2, (th-im.height)//2))
    return c


def missing(tw, th):
    c = Image.new("RGB", (tw, th), (40, 30, 30))
    ImageDraw.Draw(c).text((tw//2-40, th//2-8), "— no data —", font=F_SM, fill=(200, 160, 160))
    return c


def compose(frames, fi, name, show_gt):
    ncol = len(COL_VIEWS)
    grid_w = LABEL_W + ncol*TW + (ncol-1)*GAP
    bw, bh = int(TW*BANNER_SCALE), int(TH*BANNER_SCALE)
    y_col = bh + 34; y_pred = y_col + COLHDR; y_gt = y_pred + TH + GAP + 20
    H = (y_gt + TH + 14) if show_gt else (y_pred + TH + 14)
    W = max(grid_w, bw + 360) + 2*GAP
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
    d.text((tx, 60), f"frame {fi+1:02d}/{N_FRAMES}", font=F_SM, fill=(170, 170, 170))
    d.text((tx, 96), "WM PRED", font=F_MD, fill=PRED_C)
    if show_gt:
        d.text((tx+92, 96), "= generated", font=F_SM, fill=(170, 170, 170))
        d.text((tx, 116), "REAL GT", font=F_MD, fill=GT_C)
        d.text((tx+92, 116), "= ground truth", font=F_SM, fill=(170, 170, 170))
    else:
        d.text((tx+92, 96), "= surround views generated from front only", font=F_SM, fill=(170, 170, 170))

    d.text((GAP, y_pred+TH//2-20), "WM", font=F_LG, fill=PRED_C)
    d.text((GAP, y_pred+TH//2), "PRED", font=F_LG, fill=PRED_C)
    if show_gt:
        d.text((GAP, y_gt+TH//2-20), "REAL", font=F_LG, fill=GT_C)
        d.text((GAP, y_gt+TH//2), "GT", font=F_LG, fill=GT_C)

    for c, v in enumerate(COL_VIEWS):
        x = LABEL_W + c*(TW+GAP)
        d.text((x, y_col+4), NICE[v], font=F_SM, fill=FG)
        pf = fr(f"pred_{v}")
        canvas.paste(fit(pf, TW, TH) if pf is not None else missing(TW, TH), (x, y_pred))
        d.rectangle([x, y_pred, x+TW, y_pred+TH], outline=PRED_C, width=2)
        if show_gt:
            gf = fr(f"gt_{v}")
            canvas.paste(fit(gf, TW, TH) if gf is not None else missing(TW, TH), (x, y_gt))
            d.rectangle([x, y_gt, x+TW, y_gt+TH], outline=GT_C, width=2)
    return np.asarray(canvas)


def write_mp4(path, frames):
    frames = np.stack(frames)
    for codec in ("libx264", "mpeg4"):
        try: iio.imwrite(path, frames, plugin="pyav", codec=codec, fps=FPS); return codec
        except Exception: continue
    iio.imwrite(str(path).replace(".mp4", ".gif"), frames, plugin="pillow", duration=1000//FPS, loop=0)
    return "gif"


def find_clips(renders):
    """return {name: clip_dir} for every dir under renders that has pred_ views."""
    out = {}
    for p in glob.glob(str(renders / "**" / "pred_cross_left.mp4"), recursive=True):
        cd = Path(p).parent
        # name = the uuid dir above clip_000000 if present, else the clip dir name
        name = cd.parent.name if cd.name.startswith("clip_") else cd.name
        out[name] = cd
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=None, help="parallel dir of <uuid>/gt_<view>.mp4 + front_input.mp4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-gt", action="store_true")
    args = ap.parse_args()
    show_gt = args.with_gt  # GT may sit next to preds (cached) or in --gt (bundled)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = find_clips(args.renders)
    if not clips:
        raise SystemExit(f"no pred_*.mp4 found under {args.renders}")
    print(f"found {len(clips)} clips")

    cards = []
    for k, (name, cd) in enumerate(clips.items(), 1):
        frames = {"front": load_vid(cd / "front_input.mp4")}
        for v in COL_VIEWS:
            frames[f"pred_{v}"] = load_vid(cd / f"pred_{v}.mp4")
        if show_gt:
            # prefer GT sitting next to the preds (inference_cached writes it there
            # for free); fall back to the bundled --gt dir keyed by clip name.
            gdir = args.gt / name if args.gt is not None else None
            for v in COL_VIEWS:
                g = load_vid(cd / f"gt_{v}.mp4")
                if g is None and gdir is not None:
                    g = load_vid(gdir / f"gt_{v}.mp4")
                frames[f"gt_{v}"] = g
            if frames["front"] is None and gdir is not None:
                frames["front"] = load_vid(gdir / "front_input.mp4")
        T = min(max([len(a) for a in frames.values() if a is not None] + [1]), N_FRAMES)
        seq = [compose(frames, fi, name, show_gt) for fi in range(T)]
        Image.fromarray(seq[T//2]).save(args.out / f"{k:02d}_{name[:16]}.png")
        codec = write_mp4(str(args.out / f"{k:02d}_{name[:16]}.mp4"), seq)
        cards.append((f"{k:02d}_{name[:16]}", name))
        print(f"  [{k}/{len(clips)}] {name[:16]}  +mp4({codec})")

    title = "OpenLongTail demo — front-only → surround synthesis"
    sub = ("Each clip: the world-model sees <b>only the front camera</b> (top banner) and generates the "
           "5 side/rear rig views." + (" <span style='color:#ffc43c'>WM PRED</span> over "
           "<span style='color:#5ac8ff'>REAL GT</span>, aligned by column." if show_gt else
           " Surround views generated purely from the front video."))
    card_html = "\n".join(
        f'<div class="card"><div class="lbl">{n}</div>'
        f'<video src="{fn}.mp4" controls loop muted playsinline preload="none" poster="{fn}.png"></video></div>'
        for fn, n in cards)
    (args.out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>{title}</title>
<style>body{{margin:0;background:#0e0e10;color:#eee;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:24px 28px 4px;font-size:22px}}.sub{{color:#9a9a9a;margin:0 28px 16px;max-width:1000px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:16px;padding:8px 28px 32px}}
.card{{background:#191920;border:1px solid #24242e;border-radius:10px;overflow:hidden}}
.card video{{width:100%;display:block;background:#000}}.lbl{{padding:8px 12px;color:#ffc43c;font-family:monospace;font-size:13px}}</style>
<h1>{title}</h1><div class="sub">{sub}</div><div class="grid">{card_html}</div>""")
    print(f"\nwrote {args.out/'index.html'}  ({len(cards)} clips)")


if __name__ == "__main__":
    main()
