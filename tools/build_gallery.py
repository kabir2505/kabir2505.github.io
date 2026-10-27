#!/usr/bin/env python3
"""Compress trip photos/videos into web-sized assets + a manifest for gallery.js.

usage: python3 tools/build_gallery.py "<source dir>" [out dir]

  photos -> thumb (640px long side webp, ~35KB) + full (1600px long side webp, ~200KB)
  videos -> H.264 720p mp4 (+faststart), poster thumb + poster full;
            the featured clip also gets a silent 480p preview for autoplay
  subfolders become groups (e.g. food/); already-built files are skipped,
  so re-run it after adding new pictures.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).expanduser()
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / 'static' / 'ijcai')

FEATURED = {'charles bridge video'}            # source stems shown as the highlight
CAPTIONS = {                                   # stem -> caption (spellings checked; adds places)
    'charles bridge video': 'Charles Bridge, Prague',
    'moritzburg': 'Moritzburg Castle, near Dresden',
    'moritzberg video': 'Moritzburg Castle, near Dresden',
    'atop berlin cathedral': 'Atop the Berlin Cathedral',
    'berlin bradenburg area': 'Brandenburg Gate area, Berlin',
    'berlin cathedral': 'Berlin Cathedral',
    'bremen walk': 'A walk through Bremen',
    'bremen': 'Bremen',
    'dresden biking.TS': 'Biking in Dresden',
    'ijcai banquet  Weser Stadium ': 'IJCAI banquet at the Weserstadion, Bremen',
    'ijcai div & inclusion lunch': 'IJCAI Diversity & Inclusion lunch',
    'ijcai run': 'IJCAI run',
    'ijcai student reception': 'IJCAI student reception',
    'late nigh bremen schnoor quarter walk with ai for healthcare folks':
        'Late-night walk through Bremen’s Schnoor quarter with the AI for Healthcare folks',
    'strahov monastery brewery 2': 'Strahov Monastery Brewery, Prague',
    'Strahov Monastery Brewery': 'Strahov Monastery Brewery, Prague',
    'biryani berlin': 'Biryani in Berlin',
    "budha's bowl ma bremen schlata": 'Buddha bowl at MA, Schlachte, Bremen',
    'dresden_vietnamese': 'Vietnamese food in Dresden',
    'dresden_vietnamese 2': 'Vietnamese food in Dresden',
    'ijcai div & inclusion ice cream': 'Ice cream at the IJCAI Diversity & Inclusion lunch',
    'ijcai div and inclusion main course': 'Main course at the IJCAI Diversity & Inclusion lunch',
    'italian margarita pizza': 'Italian Margherita pizza',
    'monster ruby red': 'Monster Ultra Fantasy Ruby Red',
    'PXL_20260820_180856068': '',
}
IMAGES = {'.jpg', '.jpeg', '.png', '.heic', '.webp'}
VIDEOS = {'.mp4', '.mov', '.m4v'}


def run(*cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def slug(path):
    s = re.sub(r'[^a-z0-9]+', '-', path.stem.lower()).strip('-')
    return s or 'item'


def norm(stem):
    return re.sub(r'\s+', ' ', stem).strip().lower()


def caption(stem):
    known = {norm(k): v for k, v in CAPTIONS.items()}
    if norm(stem) in known:
        return known[norm(stem)]
    return re.sub(r'[_\s]+', ' ', stem).strip().capitalize()


def size(path):
    out = subprocess.run(['sips', '-g', 'pixelWidth', '-g', 'pixelHeight', str(path)],
                         capture_output=True, text=True).stdout
    w = int(re.search(r'pixelWidth: (\d+)', out).group(1))
    h = int(re.search(r'pixelHeight: (\d+)', out).group(1))
    return w, h


def stale(dst, src):
    return not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


def webp(src, dst, *, long_side, q):
    """resize with sips (→ temp jpeg), then encode webp with cwebp"""
    tmp = dst.with_suffix('.tmp.jpg')
    run('sips', '-s', 'format', 'jpeg', '-s', 'formatOptions', '92', '-Z', str(long_side), str(src), '--out', str(tmp))
    run('cwebp', '-quiet', '-q', str(q), '-m', '6', '-metadata', 'none', str(tmp), '-o', str(dst))
    tmp.unlink()


def build_image(src, sl, group):
    thumb, full = OUT / 'thumb' / f'{sl}.webp', OUT / 'full' / f'{sl}.webp'
    if stale(thumb, src):
        webp(src, thumb, long_side=640, q=64)
    if stale(full, src):
        webp(src, full, long_side=1600, q=74)
    w, h = size(full)
    return {'type': 'image', 'thumb': rel(thumb), 'full': rel(full), 'w': w, 'h': h}


def build_video(src, sl, group, featured):
    mp4 = OUT / 'video' / f'{sl}.mp4'
    poster_full, thumb = OUT / 'full' / f'{sl}.webp', OUT / 'thumb' / f'{sl}.webp'
    if stale(mp4, src):
        # ffmpeg applies the rotation side-data itself; cap at 720p / 30fps
        vf = "scale='if(gt(iw,ih),-2,720)':'if(gt(iw,ih),720,-2)',fps=30"
        run('ffmpeg', '-y', '-v', 'error', '-i', str(src), '-map', '0:v:0', '-map', '0:a:0?',
            '-vf', vf, '-c:v', 'libx264', '-preset', 'slow', '-crf', '27' if featured else '29',
            '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k',
            '-movflags', '+faststart', str(mp4))
    if stale(poster_full, mp4):
        frame = poster_full.with_suffix('.tmp.png')
        run('ffmpeg', '-y', '-v', 'error', '-ss', '1', '-i', str(mp4), '-frames:v', '1', str(frame))
        run('cwebp', '-quiet', '-q', '74', '-m', '6', str(frame), '-o', str(poster_full))
        frame.unlink()
    if stale(thumb, poster_full):
        webp(poster_full, thumb, long_side=640, q=64)
    w, h = size(poster_full)
    item = {'type': 'video', 'src': rel(mp4), 'thumb': rel(thumb), 'poster': rel(poster_full), 'w': w, 'h': h}
    if featured:
        # a light, silent 480p loop for the autoplaying highlight; the full clip loads only on demand
        preview = OUT / 'video' / f'{sl}-preview.mp4'
        if stale(preview, mp4):
            vf = "scale='if(gt(iw,ih),-2,480)':'if(gt(iw,ih),480,-2)',fps=24"
            run('ffmpeg', '-y', '-v', 'error', '-i', str(mp4), '-an', '-vf', vf, '-c:v', 'libx264', '-preset', 'slow',
                '-crf', '31', '-profile:v', 'high', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(preview))
        item['preview'] = rel(preview)
    return item


def rel(p):
    return p.relative_to(OUT.parent.parent).as_posix()


def main():
    for d in ('thumb', 'full', 'video'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    items = []
    files = sorted(p for p in SRC.rglob('*') if p.is_file() and p.suffix.lower() in IMAGES | VIDEOS)
    for src in files:
        group = src.parent.name if src.parent != SRC else 'trip'
        sl = slug(src) if group == 'trip' else f'{slug(src.parent)}-{slug(src)}'
        featured = norm(src.stem) in FEATURED
        print(f'· {src.relative_to(SRC)}', flush=True)
        if src.suffix.lower() in VIDEOS:
            item = build_video(src, sl, group, featured)
        else:
            item = build_image(src, sl, group)
        item.update(caption=caption(src.stem), group=group, featured=featured)
        items.append(item)
    # highlight first, then trip videos, trip photos, other groups
    items.sort(key=lambda i: (not i['featured'], i['group'] != 'trip', i['type'] != 'video'))
    (OUT / 'gallery.json').write_text(json.dumps(items, indent=1, ensure_ascii=False))
    total = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    print(f'{len(items)} items -> {OUT} ({total / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
