import numpy as np
from pathlib import Path
import pandas as pd

import numpy as np
from pathlib import Path
import pandas as pd

from collections import Counter

def analyze_csv_masks(local_dir: str, splits=('train','val','split')):
    """
    Analyze *all* mask CSVs under train/mask and val/mask.

    Args:
      local_dir: root folder of the Bluesky dataset snapshot
      splits: which sub-folders to scan (default 'train' and 'val')

    Returns:
      info: dict mapping split→filename→{shape, uniques, counts, percents}
    """
    local_dir = Path(local_dir)
    info = {}

    for split in splits:
        mask_dir = local_dir / split / 'mask'
        csv_files = sorted(mask_dir.glob('*.csv'))
        print(f"\n=== {split.upper()} MASKS ({len(csv_files)} files) ===")

        split_info = {}
        for csv_path in csv_files:
            # 1) load
            df = pd.read_csv(csv_path, header=None)
            h, w = df.shape
            arr = df.values

            # 2) unique values & counts
            uniques, counts = np.unique(arr, return_counts=True)
            counts = dict(zip(uniques.tolist(), counts.tolist()))
            percents = {k: v / (h*w) * 100 for k, v in counts.items()}

            # 3) print summary
            print(f"{csv_path.name}\t→ shape={h}×{w}")
            print(f"  uniques: {uniques.tolist()}")
            for cls in uniques:
                print(f"    {int(cls)}: {counts[int(cls)]} px ({percents[cls]:.1f}%)")

            # 4) store
            split_info[csv_path.name] = {
                'shape': (h, w),
                'uniques': uniques.tolist(),
                'counts': counts,
                'percents': percents,
            }

        info[split] = split_info

    return info

def analyze_csv_masks_overview(local_dir: str, splits=('train','val')):
    """
    Summarize all mask CSVs under train/mask and val/mask.

    Prints:
      - How many mask files of each (h×w) shape
      - Total pixel count and overall percentage per class value
    """
    local_dir = Path(local_dir)

    for split in splits:
        mask_dir = local_dir / split / 'mask'
        csv_files = sorted(mask_dir.glob('*.csv'))

        shape_counter = Counter()
        class_counter = Counter()
        total_pixels = 0

        for csv_path in csv_files:
            arr = pd.read_csv(csv_path, header=None).values
            h, w = arr.shape
            shape_counter[(h, w)] += 1

            uniques, counts = np.unique(arr, return_counts=True)
            for cls, cnt in zip(uniques, counts):
                class_counter[int(cls)] += int(cnt)
                total_pixels += int(cnt)

        print(f"\n=== {split.upper()} MASKS SUMMARY ===")
        # Shape distribution
        print("Mask shapes (rows×cols):")
        for shape, freq in shape_counter.items():
            print(f"  {shape[0]}×{shape[1]}:\t{freq} files")

        # Class pixel counts
        print("\nOverall class distribution:")
        for cls, cnt in sorted(class_counter.items()):
            pct = cnt / total_pixels * 100
            print(f"  Class {cls}:\t{cnt:,} pixels ({pct:.1f}%)")
