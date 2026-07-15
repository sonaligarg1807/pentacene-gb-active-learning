"""
subset_xyz.py

Extracts N randomly sampled frames from a multi-frame .xyz file
(frames have variable atom counts). Uses a fixed random seed
for reproducibility.

HOW TO RUN:
    python scripts/subset_xyz.py input.xyz output.xyz N [seed]
"""

import sys
import random
from pathlib import Path


def index_frames(input_path):
    """Return a list of (byte_offset, n_atoms) for each frame."""
    frame_offsets = []
    with open(input_path, "r") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            frame_offsets.append((offset, n_atoms))

            f.readline()  # comment line
            for _ in range(n_atoms):
                f.readline()

    return frame_offsets


def main():
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    n_frames = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

    frame_offsets = index_frames(input_path)
    total_frames = len(frame_offsets)

    rng = random.Random(seed)
    n_frames = min(n_frames, total_frames)
    chosen_indices = sorted(rng.sample(range(total_frames), n_frames))

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for idx in chosen_indices:
            offset, n_atoms = frame_offsets[idx]
            fin.seek(offset)

            fout.write(fin.readline())  # n_atoms line
            fout.write(fin.readline())  # comment line
            for _ in range(n_atoms):
                fout.write(fin.readline())

    print(f"Wrote {n_frames} randomly sampled frames (seed={seed}) "
          f"out of {total_frames} total to {output_path}")


if __name__ == "__main__":
    main()