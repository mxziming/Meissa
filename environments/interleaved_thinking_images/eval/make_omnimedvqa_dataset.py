#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build evaluation JSON files for OmniMedVQA.

OmniMedVQA directory structure (--omnimedvqa_dir):
    OmniMedVQA/
    ├── Images/
    │   ├── ACRIMA/
    │   │   ├── Im002_ACRIMA.png
    │   │   └── ...
    │   └── ...
    └── QA_information/
        ├── Open-access/
        │   ├── ACRIMA.json
        │   └── ...
        └── Restricted-access/
            ├── AIDA.json
            └── ...

Each QA JSON item:
    {
        "dataset": "Covid CT",
        "question_id": "Covid CT_0082",
        "question_type": "Anatomy Identification",
        "question": "What anatomical area is shown in this picture?",
        "gt_answer": "Chest region.",
        "image_path": "Images/Covid CT/CT_COVID/...",   # relative to omnimedvqa_dir
        "option_A": "Upper arm region",
        "option_B": "Chest region.",
        "option_C": "Leg region",
        "option_D": "Shoulder and upper back region",
        "modality_type": "CT(Computed Tomography)"
    }

Output files written to --out_dir:
    omnimedvqa_all.json          all items with existing images
    omnimedvqa_yesno.json        yes/no subset
    omnimedvqa_open_ended.json   non-yes/no subset
    omnimedvqa_stats.json        counts per dataset / modality / question_type

Usage:
    # Open-access only (default)
    python make_omnimedvqa_dataset.py \
        --omnimedvqa_dir /data/OmniMedVQA \
        --out_dir /data/OmniMedVQA/processed

    # Include restricted-access (images must already be placed at their paths)
    python make_omnimedvqa_dataset.py \
        --omnimedvqa_dir /data/OmniMedVQA \
        --out_dir /data/OmniMedVQA/processed \
        --include_restricted

    # Filter to a single modality
    python make_omnimedvqa_dataset.py \
        --omnimedvqa_dir /data/OmniMedVQA \
        --out_dir /data/OmniMedVQA/processed \
        --modality "CT(Computed Tomography)"
"""

import os
import json
import argparse
import collections
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _dump_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(obj):>6} items → {path}")


def _norm_yesno(s: str) -> Optional[str]:
    """Return 'yes'/'no' if the answer is a yes/no variant, else None."""
    t = s.strip().lower().rstrip(".")
    if t in ("yes", "y"):
        return "yes"
    if t in ("no", "n"):
        return "no"
    return None


def _format_question(item: Dict[str, Any], include_options: bool) -> str:
    """
    Return the question string, optionally with MCQ options appended.
    Only appends options when at least option_A is non-empty.
    """
    q = str(item.get("question", "")).strip()

    if not include_options:
        return q

    options = []
    for key in ("option_A", "option_B", "option_C", "option_D"):
        val = item.get(key)
        if val and str(val).strip():
            letter = key[-1]  # A / B / C / D
            options.append(f"{letter}. {str(val).strip()}")

    if options:
        q = q + "\n" + "\n".join(options)

    return q


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load_qa_jsons(qa_dir: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Load all *.json files under qa_dir.
    Returns list of (source_file_stem, item_dict).
    """
    records: List[Tuple[str, Dict[str, Any]]] = []
    if not os.path.isdir(qa_dir):
        return records

    for fname in sorted(os.listdir(qa_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(qa_dir, fname)
        stem = os.path.splitext(fname)[0]
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to load {fpath}: {e}")
            continue

        if isinstance(data, list):
            for item in data:
                records.append((stem, item))
        elif isinstance(data, dict):
            # some files may be a single object
            records.append((stem, data))
        else:
            print(f"  [WARN] Unexpected JSON type in {fpath}: {type(data)}")

    return records


# ---------------------------------------------------------------------------
# processing
# ---------------------------------------------------------------------------

def process_records(
    records: List[Tuple[str, Dict[str, Any]]],
    omnimedvqa_dir: str,
    include_options: bool,
    modality_filter: Optional[str],
    check_images: bool,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Returns:
        all_items, yesno_items, open_ended_items, stats
    """
    all_items: List[Dict] = []
    yesno_items: List[Dict] = []
    open_ended_items: List[Dict] = []

    stats: Dict = {
        "total_raw": len(records),
        "skipped_no_image": 0,
        "skipped_modality_filter": 0,
        "skipped_missing_fields": 0,
        "included": 0,
        "yesno": 0,
        "open_ended": 0,
        "by_dataset": collections.Counter(),
        "by_modality": collections.Counter(),
        "by_question_type": collections.Counter(),
    }

    for _stem, item in records:
        # --- required fields ---
        question_raw = item.get("question")
        gt_answer = item.get("gt_answer")
        image_path_rel = item.get("image_path")

        if not question_raw or gt_answer is None or not image_path_rel:
            stats["skipped_missing_fields"] += 1
            continue

        # --- modality filter ---
        modality = str(item.get("modality_type", "")).strip()
        if modality_filter and modality_filter.lower() not in modality.lower():
            stats["skipped_modality_filter"] += 1
            continue

        # --- image path resolution ---
        # image_path_rel is relative to omnimedvqa_dir
        # e.g. "Images/Covid CT/CT_COVID/foo.png"
        abs_image_path = os.path.join(omnimedvqa_dir, image_path_rel)
        abs_image_path = os.path.normpath(abs_image_path)

        if check_images and not os.path.isfile(abs_image_path):
            stats["skipped_no_image"] += 1
            continue

        # --- build output record ---
        question_text = _format_question(item, include_options)
        answer_text = str(gt_answer).strip()

        out = {
            "question": question_text,
            "answer": answer_text,
            "image_path": abs_image_path,
            # metadata kept for downstream filtering / analysis
            "question_id": str(item.get("question_id", "")),
            "question_type": str(item.get("question_type", "")),
            "modality_type": modality,
            "dataset": str(item.get("dataset", "")),
        }

        all_items.append(out)
        stats["included"] += 1
        stats["by_dataset"][out["dataset"]] += 1
        stats["by_modality"][modality] += 1
        stats["by_question_type"][out["question_type"]] += 1

        # --- yes/no vs open-ended split ---
        yn = _norm_yesno(answer_text)
        if yn is not None:
            yesno_out = dict(out)
            yesno_out["label"] = yn
            yesno_items.append(yesno_out)
            stats["yesno"] += 1
        else:
            open_ended_items.append(out)
            stats["open_ended"] += 1

    # convert Counter to plain dict for JSON serialisation
    stats["by_dataset"] = dict(stats["by_dataset"].most_common())
    stats["by_modality"] = dict(stats["by_modality"].most_common())
    stats["by_question_type"] = dict(stats["by_question_type"].most_common())

    return all_items, yesno_items, open_ended_items, stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build evaluation JSON files for OmniMedVQA."
    )
    ap.add_argument(
        "--omnimedvqa_dir", type=str, required=True,
        help="Root directory of OmniMedVQA (contains Images/ and QA_information/)."
    )
    ap.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory for JSON files. Default: <omnimedvqa_dir>/processed."
    )
    ap.add_argument(
        "--include_restricted", action="store_true",
        help="Also load QA_information/Restricted-access/ entries. "
             "Images must already exist at their resolved paths."
    )
    ap.add_argument(
        "--no_check_images", action="store_true",
        help="Skip the image-existence check (include items even if image file is missing)."
    )
    ap.add_argument(
        "--no_options", action="store_true",
        help="Do NOT append MCQ options (A/B/C/D) to the question text."
    )
    ap.add_argument(
        "--modality", type=str, default=None,
        help="Filter to items whose modality_type contains this string (case-insensitive). "
             "E.g. --modality CT  or  --modality 'Chest X-Ray'"
    )
    args = ap.parse_args()

    omnimedvqa_dir = os.path.abspath(args.omnimedvqa_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(omnimedvqa_dir, "processed")
    include_options = not args.no_options
    check_images = not args.no_check_images

    _ensure_dir(out_dir)

    # --- load QA JSONs ---
    open_access_dir = os.path.join(omnimedvqa_dir, "QA_information", "Open-access")
    restricted_dir = os.path.join(omnimedvqa_dir, "QA_information", "Restricted-access")

    print(f"Loading open-access QA from: {open_access_dir}")
    records = _load_qa_jsons(open_access_dir)
    print(f"  loaded {len(records)} items from open-access")

    if args.include_restricted:
        print(f"Loading restricted-access QA from: {restricted_dir}")
        restricted = _load_qa_jsons(restricted_dir)
        records.extend(restricted)
        print(f"  loaded {len(restricted)} items from restricted-access")

    print(f"\nTotal raw records: {len(records)}")

    # --- process ---
    print("\nProcessing records...")
    all_items, yesno_items, open_ended_items, stats = process_records(
        records=records,
        omnimedvqa_dir=omnimedvqa_dir,
        include_options=include_options,
        modality_filter=args.modality,
        check_images=check_images,
    )

    # --- write outputs ---
    suffix = f"_{args.modality.replace(' ', '_').replace('(', '').replace(')', '')}" if args.modality else ""
    print("\nWriting output files:")
    _dump_json(os.path.join(out_dir, f"omnimedvqa{suffix}_all.json"), all_items)
    _dump_json(os.path.join(out_dir, f"omnimedvqa{suffix}_yesno.json"), yesno_items)
    _dump_json(os.path.join(out_dir, f"omnimedvqa{suffix}_open_ended.json"), open_ended_items)

    stats_path = os.path.join(out_dir, f"omnimedvqa{suffix}_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # --- summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Raw records loaded    : {stats['total_raw']}")
    print(f"  Skipped (no image)    : {stats['skipped_no_image']}")
    print(f"  Skipped (modality)    : {stats['skipped_modality_filter']}")
    print(f"  Skipped (missing fld) : {stats['skipped_missing_fields']}")
    print(f"  Included              : {stats['included']}")
    print(f"    ├─ yes/no           : {stats['yesno']}")
    print(f"    └─ open-ended       : {stats['open_ended']}")
    print(f"\nTop modalities:")
    for mod, cnt in list(stats["by_modality"].items())[:8]:
        print(f"  {cnt:>6}  {mod}")
    print(f"\nTop question types:")
    for qt, cnt in list(stats["by_question_type"].items())[:8]:
        print(f"  {cnt:>6}  {qt}")
    print(f"\nOutput dir: {out_dir}")
    print(f"Stats file: {stats_path}")


if __name__ == "__main__":
    main()
