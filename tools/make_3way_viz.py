#!/usr/bin/env python3
"""Build a 3-WAY comparison gallery: RAW-from-scratch vs CACHED vs REAL-GT.

Each clip shows the front-camera banner (the only real input) over three
column-aligned rows of the 5 surround rig views:

  RAW   (amber) — generated end-to-end from the front mp4 alone:
                  DepthCrafter depth -> MapAnything ego-pose -> Qwen caption
                  -> live-warp -> WM generate.  Nothing pre-baked.
  CACHED(green) — the released tier: WM generate from the pre-built latent /
                  warp cache (bit-exact to the paper demos).
  REAL  (cyan)  — the true ground-truth side/rear cameras.

Clips are matched across the three sources by the 8-char uuid prefix, because
the raw path writes chunk_900/nexar_<8char>/clip_000000 while cached/GT use the
full uuid dir.

  python tools/make_3way_viz.py \
      --raw    demo/outputs/raw_e2e/renders \
      --cached demo/outputs/cached_renders \
      --gt     demo/gt \
      --out    demo/outputs/gallery_3way
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
GAP, LABEL_W, COLHDR = 8, 128, 26
BANNER_SCALE = 1.7
FPS, N_FRAMES = 10, 41
BG, FG = (14, 14, 16), (238, 238, 238)
RAW_C, CACHED_C, GT_C = (255, 196, 60), (120, 230, 140), (90, 200, 255)

# rows to render, top-to-bottom: (key-prefix, big label lines, colour)
ROWS = [("raw", ["RAW", "e2e"], RAW_C),
        ("cached", ["CACHED"], CACHED_C),
        ("gt", ["REAL", "GT"], GT_C)]


def _font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()


F_LG, F_MD, F_SM = _font(20), _font(15), _font(13)


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


def compose(frames, fi, name, caption):
    ncol = len(COL_VIEWS)
    grid_w = LABEL_W + ncol*TW + (ncol-1)*GAP
    bw, bh = int(TW*BANNER_SCALE), int(TH*BANNER_SCALE)
    y_col = bh + 34
    row_h = TH + GAP + 20
    y_rows = [y_col + COLHDR + r*row_h for r in range(len(ROWS))]
    H = y_rows[-1] + TH + 14
    W = max(grid_w, bw + 380) + 2*GAP
    W += W & 1; H += H & 1
    canvas = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(canvas)

    def fr(key):
        a = frames.get(key)
        return None if a is None or len(a) == 0 else a[min(fi, len(a)-1)]

    # front banner
    f0 = fr("front"); bx = GAP
    canvas.paste(fit(f0, bw, bh) if f0 is not None else missing(bw, bh), (bx, 30))
    d.rectangle([bx, 30, bx+bw, 30+bh], outline=(90, 90, 90), width=1)
    d.text((bx, 6), "FRONT CAMERA  —  the ONLY real input the world-model sees", font=F_MD, fill=FG)

    tx = bx + bw + 18
    d.text((tx, 34), f"clip  {name[:22]}", font=F_MD, fill=FG)
    d.text((tx, 60), f"frame {fi+1:02d}/{N_FRAMES}", font=F_SM, fill=(170, 170, 170))
    d.text((tx, 92),  "RAW e2e",  font=F_MD, fill=RAW_C)
    d.text((tx+96, 92),  "= DepthCrafter+MapAnything+Qwen, from front mp4", font=F_SM, fill=(170, 170, 170))
    d.text((tx, 112), "CACHED",   font=F_MD, fill=CACHED_C)
    d.text((tx+96, 112), "= released latent/warp cache (bit-exact)", font=F_SM, fill=(170, 170, 170))
    d.text((tx, 132), "REAL GT",  font=F_MD, fill=GT_C)
    d.text((tx+96, 132), "= true side/rear cameras", font=F_SM, fill=(170, 170, 170))
    if caption:
        # wrap the caption under the legend
        words = caption.split(); line = ""; yy = 158
        for w in words:
            if len(line) + len(w) + 1 > 58:
                d.text((tx, yy), line, font=F_SM, fill=(140, 140, 150)); yy += 17; line = w
            else:
                line = (line + " " + w).strip()
        if line:
            d.text((tx, yy), line, font=F_SM, fill=(140, 140, 150))

    # column headers
    for c, v in enumerate(COL_VIEWS):
        x = LABEL_W + c*(TW+GAP)
        d.text((x, y_col+4), NICE[v], font=F_SM, fill=FG)

    # the three rows
    for (prefix, lbl, col), y in zip(ROWS, y_rows):
        for li, ln in enumerate(lbl):
            d.text((GAP, y+TH//2-20+li*20), ln, font=F_LG, fill=col)
        for c, v in enumerate(COL_VIEWS):
            x = LABEL_W + c*(TW+GAP)
            f = fr(f"{prefix}_{v}")
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


def _prefix(name: str) -> str:
    """8-char uuid prefix used to match across sources (strip nexar_ etc.)."""
    m = re.search(r"[0-9a-f]{8}", name)
    return m.group(0) if m else name[:8]


def index_source(root: Path):
    """{prefix: clip_dir} for every dir under root that has pred_ (or gt_) views."""
    out = {}
    if root is None:
        return out
    for pat in ("pred_cross_left.mp4", "gt_cross_left.mp4"):
        for p in glob.glob(str(root / "**" / pat), recursive=True):
            cd = Path(p).parent
            name = cd.parent.name if cd.name.startswith("clip_") else cd.name
            out.setdefault(_prefix(name), cd)
    return out


def read_caption(raw_cd: Path):
    """best-effort: pull the Qwen caption recorded in the raw clip's done.json."""
    import json
    for cand in raw_cd.glob("*.json"):
        try:
            j = json.loads(cand.read_text())
            if isinstance(j, dict) and j.get("caption"):
                return j["caption"]
        except Exception:
            pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True, help="raw-from-scratch renders root")
    ap.add_argument("--cached", type=Path, required=True, help="cached renders root")
    ap.add_argument("--gt", type=Path, required=True, help="ground-truth root (<uuid>/gt_<view>.mp4)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_idx = index_source(args.raw)
    cached_idx = index_source(args.cached)
    gt_idx = index_source(args.gt)
    keys = sorted(set(raw_idx) | set(cached_idx))
    if not keys:
        raise SystemExit(f"no clips found under {args.raw} / {args.cached}")
    print(f"raw={len(raw_idx)} cached={len(cached_idx)} gt={len(gt_idx)}  -> {len(keys)} clips")

    cards = []
    for k, pref in enumerate(keys, 1):
        raw_cd, cached_cd, gt_cd = raw_idx.get(pref), cached_idx.get(pref), gt_idx.get(pref)
        frames = {}
        # front banner: prefer the raw clip's own front_input, else cached, else gt
        for src in (raw_cd, cached_cd, gt_cd):
            if src is not None:
                fv = load_vid(src / "front_input.mp4")
                if fv is not None:
                    frames["front"] = fv; break
        frames.setdefault("front", None)
        for v in COL_VIEWS:
            frames[f"raw_{v}"]    = load_vid(raw_cd / f"pred_{v}.mp4") if raw_cd else None
            frames[f"cached_{v}"] = load_vid(cached_cd / f"pred_{v}.mp4") if cached_cd else None
            g = load_vid(cached_cd / f"gt_{v}.mp4") if cached_cd else None
            if g is None and gt_cd is not None:
                g = load_vid(gt_cd / f"gt_{v}.mp4")
            frames[f"gt_{v}"] = g
        caption = read_caption(raw_cd) if raw_cd else ""

        T = min(max([len(a) for a in frames.values() if a is not None] + [1]), N_FRAMES)
        seq = [compose(frames, fi, pref, caption) for fi in range(T)]
        Image.fromarray(seq[T//2]).save(args.out / f"{k:02d}_{pref}.png")
        codec = write_mp4(str(args.out / f"{k:02d}_{pref}.mp4"), seq)
        cards.append((f"{k:02d}_{pref}", pref))
        have = "".join(c for c, cd in [("R", raw_cd), ("C", cached_cd), ("G", gt_cd)] if cd)
        print(f"  [{k}/{len(keys)}] {pref}  rows=[{have}]  +mp4({codec})")

    title = "OpenLongTail — RAW-from-scratch vs CACHED vs REAL GT"
    sub = ("Each clip: the world-model sees <b>only the front camera</b> (top banner) and generates the "
           "5 side/rear rig views. <span style='color:#ffc43c'>RAW e2e</span> "
           "(DepthCrafter→MapAnything→Qwen→generate, nothing pre-baked) over "
           "<span style='color:#78e68c'>CACHED</span> (released latent cache) over "
           "<span style='color:#5ac8ff'>REAL GT</span>, aligned by column.")
    card_html = "\n".join(
        f'<div class="card"><div class="lbl">{n}</div>'
        f'<video src="{fn}.mp4" controls loop muted playsinline preload="none" poster="{fn}.png"></video></div>'
        for fn, n in cards)
    (args.out / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>{title}</title>
<style>body{{margin:0;background:#0e0e10;color:#eee;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
h1{{margin:24px 28px 4px;font-size:22px}}.sub{{color:#9a9a9a;margin:0 28px 16px;max-width:1100px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:16px;padding:8px 28px 32px}}
.card{{background:#191920;border:1px solid #24242e;border-radius:10px;overflow:hidden}}
.card video{{width:100%;display:block;background:#000}}.lbl{{padding:8px 12px;color:#ffc43c;font-family:monospace;font-size:13px}}</style>
<h1>{title}</h1><div class="sub">{sub}</div><div class="grid">{card_html}</div>""")
    print(f"\nwrote {args.out/'index.html'}  ({len(cards)} clips)")


if __name__ == "__main__":
    main()
