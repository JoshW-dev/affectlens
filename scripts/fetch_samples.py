#!/usr/bin/env python3
"""Fetch the sample clips listed in examples/samples.json.

Third-party clips are *linked, never committed*: the repo stores only URLs,
and this script downloads them into examples/samples/ (gitignored) so you can
run a real extraction locally:

    python scripts/fetch_samples.py
    affectlens inventory --clips examples/samples

Direct media URLs (ending in .mp4, .wav, ...) are fetched with the standard
library. Anything else — e.g. YouTube links — is handed to yt-dlp, which must
be installed separately (`pip install yt-dlp`).

samples.json format:

    {
      "clips": [
        {"name": "sunrise", "url": "https://example.com/sunrise.mp4"},
        {"name": "talk", "url": "https://youtube.com/watch?v=...",
         "start": 30, "duration": 60}
      ]
    }

`start`/`duration` (seconds, optional) trim the download to a section; they
are honored for yt-dlp sources only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "examples" / "samples.json"
OUT_DIR = ROOT / "examples" / "samples"

DIRECT_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def is_direct_media(url: str) -> bool:
    path = url.split("?", 1)[0].split("#", 1)[0]
    return Path(path).suffix.lower() in DIRECT_SUFFIXES


def fetch_direct(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def fetch_ytdlp(clip: dict, dest_stem: Path) -> None:
    if shutil.which("yt-dlp") is None:
        sys.exit(
            f"'{clip['url']}' is not a direct media link and yt-dlp is not "
            "installed. Install it with: pip install yt-dlp"
        )
    cmd = [
        "yt-dlp",
        "-f", "mp4/bestaudio",
        "-o", str(dest_stem) + ".%(ext)s",
        clip["url"],
    ]
    if "start" in clip or "duration" in clip:
        start = float(clip.get("start", 0))
        end = start + float(clip["duration"]) if "duration" in clip else "inf"
        cmd += ["--download-sections", f"*{start}-{end}"]
    print(f"  yt-dlp {clip['url']}")
    subprocess.run(cmd, check=True)


def main() -> None:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    clips = json.loads(MANIFEST.read_text(encoding="utf-8")).get("clips", [])
    if not clips:
        print(f"No clips listed in {MANIFEST.relative_to(ROOT)} — add some URLs first.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        name, url = clip["name"], clip["url"]
        existing = list(OUT_DIR.glob(f"{name}.*"))
        if existing:
            print(f"[skip] {name} ({existing[0].name} already present)")
            continue
        print(f"[get]  {name}")
        if is_direct_media(url):
            suffix = Path(url.split("?", 1)[0]).suffix
            fetch_direct(url, OUT_DIR / f"{name}{suffix}")
        else:
            fetch_ytdlp(clip, OUT_DIR / name)
    print(f"Done. Clips are in {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
