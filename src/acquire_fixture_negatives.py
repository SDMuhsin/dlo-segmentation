#!/usr/bin/env python
"""P32 — acquire license-clean (CC0/CC-BY/CC-BY-SA/PD) close-up photos of REAL
electrical-FIXTURE SURFACES as PURE all-background hard-negatives.

KEYLESS SOURCE: Wikimedia Commons API (https://commons.wikimedia.org/w/api.php).
The API (search/imageinfo) is generous; the IMAGE host (upload.wikimedia.org) rate-
limits, but STANDARD CACHED THUMBNAILS (e.g. 480px) at ~5 s spacing download reliably
(~80-85% success). We request a fixed standard thumb width and pace downloads.

(Openverse was the first-choice keyless source but its anonymous tier is Cloudflare-
rate-limited after a small burst and token registration is human-challenge gated, so
Commons is the dependable bulk source.)

Per image we record source_url, license, query, sha256 into manifest.csv. Save into
--out-dir. NO wires composited (pure all-background negatives).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request

from PIL import Image

UA = "wire-seg-research/1.0 (CC-image research acquisition)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

QUERIES = [
    "electrical terminal block", "terminal strip wiring", "din rail terminal block",
    "screw terminal block", "feed through terminal block", "rail terminal connector",
    "power outlet socket", "electrical wall socket", "wall receptacle outlet",
    "schuko socket", "power point socket", "double socket outlet",
    "light switch wall plate", "wall switch plate electrical", "electrical switch plate",
    "circuit breaker panel", "distribution board electrical", "consumer unit breakers",
    "fuse box electrical", "breaker panel board", "electrical panelboard",
    "miniature circuit breaker", "MCB din rail", "circuit breaker row",
    "electrical junction box", "junction box wiring", "electrical conduit box",
    "electrical conduit pipe", "conduit fittings electrical", "metal conduit wiring",
    "cable trunking", "cable raceway", "cable duct wiring", "wire duct panel",
    "electrical switchgear", "low voltage switchgear", "electrical enclosure",
    "control panel wiring", "industrial control panel", "electrical contactor panel",
    "busbar electrical", "wiring terminal connector", "electrical meter box",
    "distribution panel breakers", "din rail mounted", "electrical socket strip",
    "relay terminal panel", "motor control center", "earthing terminal block",
    "fuse holder din rail", "power distribution panel", "electrical wiring panel",
    "us electrical outlet receptacle", "european power socket", "terminal block row",
]

THUMB_WIDTH = 480
IMG_INTERVAL = 5.0   # seconds between image-thumb downloads
API_DELAY = 0.5
_last_img = [0.0]


def http(url, timeout=50, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(1.0 + attempt)
    return None


def http_json(url):
    raw = http(url)
    return json.loads(raw.decode("utf-8")) if raw else None


def download_thumb(url, retries=4):
    for attempt in range(retries):
        wait = IMG_INTERVAL - (time.time() - _last_img[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            _last_img[0] = time.time()
            return data
        except urllib.error.HTTPError as e:
            _last_img[0] = time.time()
            if e.code in (429, 503):
                time.sleep(IMG_INTERVAL * (attempt + 1))
                continue
            return None
        except Exception:
            _last_img[0] = time.time()
            time.sleep(IMG_INTERVAL)
    return None


def license_ok(s):
    s = (s or "").lower()
    if not s:
        return False
    if "nc" in s.split() or "noncommercial" in s or "non-commercial" in s:
        return False
    if "noderiv" in s or s.endswith(" nd"):
        return False
    return ("cc0" in s or "public domain" in s or "cc by" in s or "cc-by" in s or "pdm" in s)


def save_image(raw, out_dir, base, min_side=200):
    try:
        Image.open(io.BytesIO(raw)).verify()
        im = Image.open(io.BytesIO(raw))
        w, h = im.size
        fmt = (im.format or "JPEG").lower()
    except Exception:
        return None
    if w < min_side or h < min_side:
        return None
    ext = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "gif": "png"}.get(fmt, "jpg")
    sha = hashlib.sha256(raw).hexdigest()
    path = os.path.join(out_dir, f"{base}.{ext}")
    with open(path, "wb") as f:
        f.write(raw)
    return path, sha, w, h


def search_titles(query, max_titles=80):
    titles, offset = [], 0
    while len(titles) < max_titles:
        p = {"action": "query", "format": "json", "list": "search",
             "srsearch": f"{query} filetype:bitmap", "srnamespace": 6,
             "srlimit": 50, "sroffset": offset}
        d = http_json(COMMONS_API + "?" + urllib.parse.urlencode(p))
        if not d:
            break
        hits = d.get("query", {}).get("search", [])
        if not hits:
            break
        titles += [h["title"] for h in hits]
        cont = d.get("continue", {}).get("sroffset")
        if cont is None:
            break
        offset = cont
        time.sleep(API_DELAY)
    return titles[:max_titles]


def imageinfo(titles):
    out = {}
    for i in range(0, len(titles), 40):
        batch = titles[i:i + 40]
        p = {"action": "query", "format": "json", "titles": "|".join(batch),
             "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": THUMB_WIDTH}
        d = http_json(COMMONS_API + "?" + urllib.parse.urlencode(p))
        if not d:
            continue
        for _, pg in d.get("query", {}).get("pages", {}).items():
            ii = pg.get("imageinfo")
            if ii:
                out[pg["title"]] = ii[0]
        time.sleep(API_DELAY)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/hardneg_fixtures_raw")
    ap.add_argument("--manifest", default="data/hardneg_fixtures_raw/manifest.csv")
    ap.add_argument("--max-total", type=int, default=800)
    ap.add_argument("--per-query", type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fields = ["filename", "source", "source_url", "image_url", "license", "license_url",
              "query", "title", "creator", "width", "height", "sha256"]
    new = not os.path.exists(args.manifest) or os.path.getsize(args.manifest) == 0
    f = open(args.manifest, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if new:
        writer.writeheader()
    seen_sha, seen_title = set(), set()
    count = 0

    print("=== Wikimedia Commons (480px cached thumbs, ~5s spacing) ===", flush=True)
    for q in QUERIES:
        if count >= args.max_total:
            break
        titles = [t for t in search_titles(q, args.per_query) if t not in seen_title]
        info = imageinfo(titles)
        got = 0
        for t in titles:
            if count >= args.max_total:
                break
            seen_title.add(t)
            ii = info.get(t)
            if not ii:
                continue
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not license_ok(lic):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            raw = download_thumb(url)
            if raw is None:
                continue
            sha = hashlib.sha256(raw).hexdigest()
            if sha in seen_sha:
                continue
            saved = save_image(raw, args.out_dir, f"wc_{count:05d}")
            if not saved:
                continue
            path, sha, w, h = saved
            seen_sha.add(sha)
            writer.writerow({
                "filename": os.path.basename(path), "source": "wikimedia_commons",
                "source_url": ii.get("descriptionurl", url), "image_url": url,
                "license": lic, "license_url": meta.get("LicenseUrl", {}).get("value", ""),
                "query": q, "title": t[:200],
                "creator": (meta.get("Artist", {}).get("value", "") or "")[:150]
                    .replace("\n", " ").replace("\r", " "),
                "width": w, "height": h, "sha256": sha,
            })
            count += 1
            got += 1
        f.flush()
        print(f"  {q!r}: titles={len(titles)} +{got}  (total {count})", flush=True)

    f.close()
    print(f"\nDONE: {count} images acquired into {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
