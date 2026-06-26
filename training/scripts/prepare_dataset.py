"""Extract face crops from FaceForensics++ (or similar) videos into a train/val/test split.

Run on Kaggle (or locally) against a downloaded/mounted copy of the dataset.

Expected input layout (FF++ "flat" Kaggle mirror, e.g. xdxd003/ff-c23):
    raw_dir/
        original/<video_id>.mp4                  e.g. 071.mp4
        Deepfakes/<target_id>_<source_id>.mp4     e.g. 071_054.mp4
        splits/train.json, val.json, test.json    (official FF++ split: lists of [target, source] id pairs)

Output layout:
    output_dir/{train,val,test}/{real,fake}/<video_id>_<frame_idx>.jpg

Usage:
    python prepare_dataset.py --raw_dir /kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23 \
        --output_dir /kaggle/working/data --real_glob "original/*.mp4" --fake_glob "Deepfakes/*.mp4" \
        --splits_dir training/configs/splits
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Make backend/app importable so face_extractor.py is the single source of truth
# shared with inference (repo_root/backend is two levels up from this file).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.detection.face_extractor import extract_face  # noqa: E402

FRAME_STRIDE = 10       # sample every Nth frame
MAX_CROPS_PER_VIDEO = 35


def sample_frames(video_path: Path, stride: int, max_crops: int):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    saved = 0
    while saved < max_crops:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            yield frame_idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            saved += 1
        frame_idx += 1
    cap.release()


def process_video(video_path: Path, out_dir: Path, video_id: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for frame_idx, frame_rgb in sample_frames(video_path, FRAME_STRIDE, MAX_CROPS_PER_VIDEO):
        crop = extract_face(frame_rgb)
        if crop is None:
            continue
        out_path = out_dir / f"{video_id}_{frame_idx}.jpg"
        if out_path.exists():
            continue  # resumable: skip already-extracted frames
        cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        written += 1
    return written


def load_split_ids(splits_dir: Path, split_name: str) -> set[str]:
    split_file = splits_dir / f"{split_name}.json"
    with open(split_file) as f:
        pairs = json.load(f)
    # FF++ split files are lists of [id_a, id_b] pairs; flatten to a set of video ids
    ids = set()
    for pair in pairs:
        if isinstance(pair, list):
            ids.update(pair)
        else:
            ids.add(pair)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--real_glob", default="original/*.mp4")
    parser.add_argument("--fake_glob", default="Deepfakes/*.mp4")
    parser.add_argument("--splits_dir", default=None, type=Path,
                         help="dir with train.json/val.json/test.json (official FF++ split lists)")
    args = parser.parse_args()

    splits = {}
    if args.splits_dir:
        for name in ("train", "val", "test"):
            splits[name] = load_split_ids(args.splits_dir, name)

    def split_for(stem: str) -> str:
        if not splits:
            return "train"  # caller must pre-split if no official splits dir given
        # Fake video stems are "<target_id>_<source_id>" (e.g. "071_054") — split membership
        # is determined by the target identity, which is the first component.
        target_id = stem.split("_")[0]
        for name, ids in splits.items():
            if target_id in ids:
                return name
        return "train"

    for label, pattern in (("real", args.real_glob), ("fake", args.fake_glob)):
        videos = sorted(args.raw_dir.glob(pattern))
        print(f"[{label}] found {len(videos)} videos")
        for video_path in videos:
            video_id = video_path.stem
            split = split_for(video_id)
            out_dir = args.output_dir / split / label
            n = process_video(video_path, out_dir, video_id)
            print(f"  {video_id} ({split}) -> {n} crops")


if __name__ == "__main__":
    main()
