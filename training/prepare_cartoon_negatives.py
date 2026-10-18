"""Licensed, source-held-out animation pool. No model is loaded by this script.

Download -> extract -> contact sheets -> explicit visual-review ledger -> lock.
Internet Archive is a public distribution mirror, not an unrestricted image crawl.
"""

import argparse, hashlib, json, subprocess, urllib.request, urllib.parse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn
from common import ROOT, save_json, sha256

BASE = ROOT / "data/v4_phase2"
REPORT = ROOT / "results/v4_phase2"
SOURCES = [
    (
        "BigBuckBunny_328",
        "BigBuckBunny_512kb.mp4",
        "train",
        "CC-BY-3.0-US",
        "Blender Foundation / Peach team",
        "round_cartoon_animals",
        2,
    ),
    (
        "CaminandesLlamigos",
        "Caminandes_ Llamigos-1080p.mp4",
        "train",
        "CC-BY-3.0",
        "Blender Foundation / Caminandes team",
        "llama_penguin_cartoon",
        1,
    ),
    (
        "CosmosLaundromatFirstCycle",
        "Cosmos Laundromat - First Cycle (1080p).mp4",
        "train",
        "CC-BY-4.0",
        "Blender Foundation / Gooseberry team",
        "anthropomorphic_sheep",
        2,
    ),
    (
        "Sintel",
        "sintel-2048-stereo_512kb.mp4",
        "val",
        "CC-BY-3.0",
        "Blender Foundation / Durian team",
        "dragon_fantasy_animation",
        3,
    ),
    (
        "wing_it",
        "wing_it.mp4",
        "test",
        "CC-BY-4.0",
        "Blender Studio / Rik Schutte",
        "round_bird_scifi_cartoon",
        3,
    ),
    (
        "sprite-fright",
        "Sprite Fright - Open Movie by Blender Studio-804p.mp4",
        "test",
        "CC-BY-4.0",
        "Blender Studio / Sprite Fright team",
        "mushroom_humanoid_cartoon",
        3,
    ),
]


def download(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    tmp.replace(path)


def hashes(path):
    im = Image.open(path).convert("RGB")
    small = im.resize((32, 32)).convert("L")
    a = dctn(np.asarray(small, dtype=float), norm="ortho")[:8, :8].flatten()
    p = a > np.median(a[1:])
    thumb = np.asarray(im.resize((16, 16)), dtype=float) / 255
    return (
        "".join("1" if b else "0" for b in p),
        thumb,
        hashlib.sha256(im.tobytes()).hexdigest(),
    )


def sheets(rows, name):
    folder = REPORT / "review_sheets"
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), 48):
        page = Image.new("RGB", (8 * 180, 6 * 135), "white")
        draw = ImageDraw.Draw(page)
        for i, r in enumerate(rows[start : start + 48]):
            im = ImageOps.contain(Image.open(ROOT / r["path"]), (176, 108))
            x = (i % 8) * 180
            y = (i // 8) * 135
            page.paste(im, (x, y))
            draw.text((x, y + 109), r["image_id"].split("/")[-1], fill="black")
        page.save(folder / f"{name}_{start//48:02d}.jpg", quality=91)


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    allrows = []
    stats = []
    for source, filename, role, license_id, credit, category, interval in SOURCES:
        folder = BASE / "sources" / source
        folder.mkdir(parents=True, exist_ok=True)
        metadata = folder / "metadata.json"
        download("https://archive.org/metadata/" + source, metadata)
        m = json.loads(metadata.read_text())
        lic = m["metadata"].get("licenseurl", "")
        if "/licenses/by/" not in lic:
            raise ValueError(("Attribution-only license expected", source, lic))
        video = folder / filename
        url = (
            "https://archive.org/download/"
            + source
            + "/"
            + urllib.parse.quote(filename)
        )
        download(url, video)
        entry = next(r for r in m["files"] if r["name"] == filename)
        if int(entry["size"]) != video.stat().st_size:
            raise ValueError("Truncated video")
        if (
            entry.get("md5")
            and hashlib.md5(video.read_bytes()).hexdigest() != entry["md5"]
        ):
            raise ValueError("Video MD5 mismatch")
        out = BASE / "frames" / source
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "complete.json").exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-threads",
                    "2",
                    "-i",
                    str(video),
                    "-vf",
                    f"fps=1/{interval},scale=480:-2",
                    "-q:v",
                    "2",
                    str(out / "%05d.jpg"),
                ],
                check=True,
            )
            save_json(
                out / "complete.json",
                {"video_sha256": sha256(video), "interval_seconds": interval},
            )
        video_sha = sha256(video)
        kept = []
        duplicates = []
        for path in sorted(out.glob("*.jpg")):
            ph, thumb, pixel = hashes(path)
            duplicate = None
            for previous in kept:
                hd = sum(a != b for a, b in zip(ph, previous["_ph"]))
                mad = float(np.abs(thumb - previous["_thumb"]).mean())
                if pixel == previous["pixel_sha256"] or (hd <= 8 and mad < 0.07):
                    duplicate = previous["image_id"]
                    break
            image_id = source + "/" + path.stem
            if duplicate:
                duplicates.append({"image_id": image_id, "duplicate_of": duplicate})
                continue
            r = {
                "image_id": image_id,
                "path": str(path.relative_to(ROOT)),
                "source": source,
                "identity_group": source + ":entire_cast",
                "original_group": source,
                "shot_group": source + ":" + str((int(path.stem) * interval) // 15),
                "category": category,
                "split": role,
                "label": 2,
                "domain": "cartoon",
                "timestamp_seconds": (int(path.stem) - 0.5) * interval,
                "download_url": url,
                "license": license_id,
                "license_url": lic,
                "attribution": credit,
                "source_metadata": str(metadata.relative_to(ROOT)),
                "source_video_sha256": video_sha,
                "sha256": sha256(path),
                "pixel_sha256": pixel,
                "phash": hex(int(ph, 2))[2:].zfill(16),
                "_ph": ph,
                "_thumb": thumb,
            }
            kept.append(r)
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in kept]
        allrows.extend(public)
        sheets(public, source)
        stats.append(
            {
                "source": source,
                "role": role,
                "frames_extracted": len(list(out.glob("*.jpg"))),
                "deduplicated_candidates": len(public),
                "removed_near_duplicates": len(duplicates),
                "license": lic,
                "video_sha256": sha256(video),
                "duplicates": duplicates,
            }
        )
        print(source, stats[-1]["frames_extracted"], len(public), flush=True)
    save_json(
        BASE / "unreviewed_manifest.json",
        {
            "stage": "unreviewed; not eligible for training/mining/test",
            "samples": allrows,
            "sources": stats,
        },
    )
    print(
        "Inspect all contact sheets, write experiments/v4_phase2/visual_review.json, then --lock.",
        flush=True,
    )


def lock():
    raw = json.loads((BASE / "unreviewed_manifest.json").read_text())
    review = json.loads((ROOT / "experiments/v4_phase2/visual_review.json").read_text())
    selected = []
    reviewed = []
    for r in raw["samples"]:
        rule = review["sources"][r["source"]]
        n = int(r["image_id"].split("/")[-1])
        ok = any(a <= n <= b for a, b in rule["keep_ranges"]) and n not in rule.get(
            "exclude_ids", []
        )
        reviewed.append(
            {
                "image_id": r["image_id"],
                "eligible": ok,
                "reviewer": review["reviewer"],
                "reason": (
                    rule["scope"]
                    if ok
                    else "empty/credits/environment-only/insufficient character evidence/graphic content"
                ),
            }
        )
        if ok:
            for category, ranges in rule.get("category_ranges", {}).items():
                if any(a <= n <= b for a, b in ranges):
                    r = {**r, "category": category}
            selected.append(
                {
                    **r,
                    "identity_review": "assistant contact-sheet verified non-Nailong/non-Naiwa; known movie cast",
                    "review_ledger": "results/v4_phase2/identity_review.json",
                    "ever_train_before_phase2": False,
                    "ever_validation_before_phase2": False,
                }
            )
    # Cross-role pHash AND actual pixel-thumbnail check. Role separation is by entire cast/movie,
    # so another crop of the same original scene is also never intentionally reassigned.
    cached = [(r, *hashes(ROOT / r["path"])[:2]) for r in selected]
    flags = []
    for i, (r, p, t) in enumerate(cached):
        for s, q, u in cached[:i]:
            if r["split"] == s["split"]:
                continue
            hd = sum(a != b for a, b in zip(p, q))
            mad = float(np.abs(t - u).mean())
            if r["pixel_sha256"] == s["pixel_sha256"] or (hd <= 10 and mad < 0.09):
                flags.append(
                    {
                        "a": r["image_id"],
                        "b": s["image_id"],
                        "phash_hamming": hd,
                        "rgb_mad": mad,
                    }
                )
    if flags:
        save_json(REPORT / "cross_split_flags.json", flags)
        raise ValueError("Resolve cross-role visual duplicate flags before locking")
    # Compare all project holdouts + original train/OE, not just file SHA.
    existing = []
    for root in [
        ROOT / "data/NaiwaOrNailong-dataset",
        ROOT / "data/hard_ood",
        ROOT / "data/hard_ood_benchmark",
        ROOT / "data/hard_negative",
        ROOT / "data/false_positive",
    ]:
        if root.exists():
            existing.extend(
                p
                for p in root.rglob("*")
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            )
    legacy = []
    for p in existing:
        h, t, pixel = hashes(p)
        legacy.append((str(p.relative_to(ROOT)), h, t, pixel))
    legacyflags = []
    for r, p, t in cached:
        for path, q, u, pixel in legacy:
            hd = sum(a != b for a, b in zip(p, q))
            if r["pixel_sha256"] == pixel or (
                hd <= 10 and float(np.abs(t - u).mean()) < 0.09
            ):
                legacyflags.append(
                    {"new": r["image_id"], "legacy": path, "phash_hamming": hd}
                )
    if legacyflags:
        save_json(REPORT / "legacy_duplicate_flags.json", legacyflags)
        raise ValueError("Resolve legacy visual duplicate flags")
    manifest = {
        "version": "v4-phase2",
        "protocol_sha256": sha256(ROOT / "experiments/v4_phase2/protocol.json"),
        "data_addendum_sha256": sha256(
            ROOT / "experiments/v4_phase2/data_protocol_addendum.json"
        ),
        "review_sha256": sha256(ROOT / "experiments/v4_phase2/visual_review.json"),
        "split_policy": "Whole movie and entire cast held out; fixed before any new inference. Test is model-blind visual content curation, not model-selection data.",
        "samples": selected,
        "sources": raw["sources"],
        "limitations": review["limitations"],
    }
    target = BASE / "cartoon_manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Manifest is already frozen; refuse overwrite")
    save_json(target, manifest)
    save_json(REPORT / "identity_review.json", reviewed)
    save_json(
        REPORT / "data_audit.json",
        {
            "manifest_sha256": sha256(target),
            "reviewed_candidates": len(reviewed),
            "eligible": len(selected),
            "counts": {
                role: sum(r["split"] == role for r in selected)
                for role in ["train", "val", "test"]
            },
            "cross_split_phash_rgb_flags": flags,
            "legacy_phash_rgb_flags": legacyflags,
            "legacy_images_compared": len(legacy),
            "identity_split_overlap": False,
            "known_duplicate_detection_limit": "pHash+thumbnail is not a proof of absence of arbitrary crops; whole film/cast grouping and manual review provide additional protection.",
            "external_human_review": False,
        },
    )
    print("LOCKED", sha256(target), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock", action="store_true")
    a = p.parse_args()
    lock() if a.lock else prepare()
