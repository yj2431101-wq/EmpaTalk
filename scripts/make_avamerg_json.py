"""Generate speaker-listener pair JSON for AvaMERG training.

Filename format: dia{dia_id}utt{utt_id}_{speaker_id}.mp4

Pairing logic:
  For each utterance N where speaker is person A, the listener is person B.
  We use person B's nearest available utterance video as the listener clip.

Output JSON (Variant A, ID-based, compatible with AvaMERGDataset):
  [{"speaker": "dia05596utt1_28", "listener": "dia05596utt2_13"}, ...]

Usage:
  python scripts/make_avamerg_json.py \
      --video_dir /mnt/data/dataset/AvaMERG/video_v5_0 \
      --output    /mnt/data/dataset/AvaMERG/train.json \
      [--val_output /mnt/data/dataset/AvaMERG/val.json] \
      [--val_ratio 0.05]
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict


FILENAME_RE = re.compile(r"^(dia\d+)(utt\d+)_(\d+)\.mp4$")


def parse_videos(video_dir: str):
    """Return list of (dia_id, utt_idx, speaker_id, stem) for every mp4 found."""
    records = []
    for fname in os.listdir(video_dir):
        m = FILENAME_RE.match(fname)
        if m is None:
            continue
        dia_id = m.group(1)            # e.g. "dia05596"
        utt_part = m.group(2)          # e.g. "utt1"
        speaker_id = m.group(3)        # e.g. "28"
        utt_idx = int(utt_part[3:])    # 1
        stem = fname[:-4]              # "dia05596utt1_28"
        records.append((dia_id, utt_idx, speaker_id, stem))
    return records


def build_pairs(records):
    """Build (speaker_stem, listener_stem) pairs from AvaMERG records.

    Strategy:
      - Group by dialogue.
      - Within each dialogue, identify the two participants.
      - For each utterance (speaker=A), find person B's nearest utterance
        (by utt_idx distance) to use as the listener clip.
    """
    # Group: dia_id -> {utt_idx: (speaker_id, stem)}
    dia_map = defaultdict(dict)
    for dia_id, utt_idx, spk_id, stem in records:
        dia_map[dia_id][utt_idx] = (spk_id, stem)

    pairs = []
    for dia_id, utts in dia_map.items():
        sorted_utts = sorted(utts.items())  # [(utt_idx, (spk_id, stem)), ...]

        # Collect all stems per participant within this dialogue
        participant_utts = defaultdict(list)  # spk_id -> sorted list of (utt_idx, stem)
        for utt_idx, (spk_id, stem) in sorted_utts:
            participant_utts[spk_id].append((utt_idx, stem))

        participants = list(participant_utts.keys())
        if len(participants) < 2:
            # Single-speaker dialogue – skip
            continue

        for utt_idx, (spk_id, spk_stem) in sorted_utts:
            # Find the other participant(s)
            other_ids = [p for p in participants if p != spk_id]
            for other_id in other_ids:
                other_options = participant_utts[other_id]
                if not other_options:
                    continue
                # Pick the nearest utterance of the other speaker
                nearest = min(other_options, key=lambda x: abs(x[0] - utt_idx))
                lis_stem = nearest[1]
                pairs.append({"speaker": spk_stem, "listener": lis_stem})

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True,
                        help="Path to AvaMERG video_v5_0/ directory")
    parser.add_argument("--output", required=True,
                        help="Output JSON path (train split)")
    parser.add_argument("--val_output", default=None,
                        help="Output JSON path for validation split (optional)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Fraction of dialogues for validation (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Scanning {args.video_dir} ...")
    records = parse_videos(args.video_dir)
    print(f"  Found {len(records)} videos")

    if args.val_output:
        # Split by dialogue to avoid leakage
        dia_ids = list({r[0] for r in records})
        random.shuffle(dia_ids)
        n_val = max(1, int(len(dia_ids) * args.val_ratio))
        val_dias = set(dia_ids[:n_val])
        train_records = [r for r in records if r[0] not in val_dias]
        val_records   = [r for r in records if r[0] in val_dias]

        train_pairs = build_pairs(train_records)
        val_pairs   = build_pairs(val_records)

        with open(args.output, "w") as f:
            json.dump(train_pairs, f, indent=2)
        with open(args.val_output, "w") as f:
            json.dump(val_pairs, f, indent=2)

        print(f"Train pairs : {len(train_pairs):>6}  -> {args.output}")
        print(f"Val   pairs : {len(val_pairs):>6}  -> {args.val_output}")
    else:
        pairs = build_pairs(records)
        with open(args.output, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Total pairs : {len(pairs):>6}  -> {args.output}")


if __name__ == "__main__":
    main()
