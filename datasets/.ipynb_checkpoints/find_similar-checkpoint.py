#!/usr/bin/env python3
"""
Find image pairs satisfying:

    np.abs(x - y).mean() < threshold

where x and y are images normalized to [0, 1].

For uint8 images loaded from disk, this is equivalent to:

    sum(abs(x_u8 - y_u8)) < threshold * 255 * num_elements

This script uses a multi-stage exact pipeline:
1. Load every image once and compute coarse block-sum features.
2. Use a KD-tree in coarse-feature L1 space to generate an exact candidate superset.
3. Apply a finer block-sum L1 lower bound to prune more candidates.
4. Re-open only the surviving candidates and compute the exact MAD.

Outputs:
- exact_pairs: every pair that truly satisfies the threshold
- connected_components: graph components induced by exact_pairs

Note:
The relation is not transitive, so "connected_components" are only a convenient grouping.
If you need maximal cliques instead, that is a different and usually more expensive problem.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find image pairs with exact MAD below a tiny threshold."
    )
    parser.add_argument("--image-dir", default="pat", help="Directory of images, default: pat")
    parser.add_argument("--suffix", default=".png", help="Image suffix, default: .png")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0 / 255.0,
        help="Threshold on normalized MAD, default: 1/255",
    )
    parser.add_argument(
        "--coarse-grid",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("H", "W"),
        help="Coarse block grid, default: 8 8",
    )
    parser.add_argument(
        "--fine-grid",
        type=int,
        nargs=2,
        default=(20, 20),
        metavar=("H", "W"),
        help="Fine block grid, default: 20 20",
    )
    parser.add_argument(
        "--output",
        default="low_mad_groups.json",
        help="Output JSON path, default: low_mad_groups.json",
    )
    return parser.parse_args()


def list_image_paths(image_dir: str, suffix: str) -> List[str]:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(suffix.lower())
    )


def load_uint8_image(path: str) -> np.ndarray:
    # with Image.open(path) as img:
    #     arr = np.asarray(img)
    arr = np.load(path)[0,0]*255
    # print(arr.shape)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    return np.ascontiguousarray(arr)


def block_sum_feature(arr: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    h, w, c = arr.shape
    if h % grid_h != 0 or w % grid_w != 0:
        raise ValueError(
            f"Image shape {(h, w)} is not divisible by grid {(grid_h, grid_w)}. "
            "Please choose a compatible grid."
        )

    bh = h // grid_h
    bw = w // grid_w
    feat = (
        arr.astype(np.int32)
        .reshape(grid_h, bh, grid_w, bw, c)
        .sum(axis=(1, 3))
        .reshape(-1)
        .astype(np.int64)
    )
    return feat


def extract_features(
    image_paths: Sequence[str],
    coarse_grid: Tuple[int, int],
    fine_grid: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    coarse_feats: List[np.ndarray] = []
    fine_feats: List[np.ndarray] = []
    image_shape = None

    for idx, path in enumerate(image_paths):
        arr = load_uint8_image(path)
        if image_shape is None:
            image_shape = arr.shape
        elif arr.shape != image_shape:
            raise ValueError(
                f"All images must have the same shape. "
                f"First shape={image_shape}, but {path} has {arr.shape}."
            )

        coarse_feats.append(block_sum_feature(arr, coarse_grid[0], coarse_grid[1]))
        fine_feats.append(block_sum_feature(arr, fine_grid[0], fine_grid[1]))

        if (idx + 1) % 500 == 0:
            print(f"Loaded features for {idx + 1}/{len(image_paths)} images")

    coarse = np.stack(coarse_feats, axis=0)
    fine = np.stack(fine_feats, axis=0)
    meta = {
        "image_shape": list(image_shape),
        "num_elements": int(np.prod(image_shape)),
    }
    return coarse, fine, meta


def candidate_pairs_from_coarse(
    coarse_feats: np.ndarray,
    raw_l1_threshold: float,
) -> np.ndarray:
    tree = cKDTree(coarse_feats.astype(np.float64))
    pairs = tree.query_pairs(r=raw_l1_threshold, p=1.0, output_type="ndarray")
    if pairs.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return pairs.astype(np.int32, copy=False)


def prune_with_fine_feature(
    pairs: np.ndarray,
    fine_feats: np.ndarray,
    raw_l1_threshold: float,
    batch_size: int = 20000,
) -> np.ndarray:
    if len(pairs) == 0:
        return pairs

    kept_batches: List[np.ndarray] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        lhs = fine_feats[batch[:, 0]]
        rhs = fine_feats[batch[:, 1]]
        lower_bound = np.abs(lhs - rhs).sum(axis=1, dtype=np.int64)
        kept = batch[lower_bound < raw_l1_threshold]
        if len(kept):
            kept_batches.append(kept)

    if not kept_batches:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(kept_batches, axis=0)


def build_exact_verifier(image_paths: Sequence[str]):
    @lru_cache(maxsize=512)
    def _load_by_index(index: int) -> np.ndarray:
        return load_uint8_image(image_paths[index])

    def verify_pair(i: int, j: int, raw_l1_threshold: float) -> bool:
        a = _load_by_index(i)
        b = _load_by_index(j)
        diff_sum = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(dtype=np.int64)
        return diff_sum < raw_l1_threshold

    return verify_pair


def exact_pairs(
    image_paths: Sequence[str],
    candidate_pairs: np.ndarray,
    raw_l1_threshold: float,
) -> List[Tuple[int, int]]:
    verify_pair = build_exact_verifier(image_paths)
    exact: List[Tuple[int, int]] = []

    for idx, (i, j) in enumerate(candidate_pairs):
        if verify_pair(int(i), int(j), raw_l1_threshold):
            exact.append((int(i), int(j)))

        if (idx + 1) % 2000 == 0:
            print(f"Exact verification: {idx + 1}/{len(candidate_pairs)} candidates")

    return exact


def connected_components(num_images: int, pairs: Sequence[Tuple[int, int]]) -> List[List[int]]:
    uf = UnionFind(num_images)
    for i, j in pairs:
        uf.union(i, j)

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(num_images):
        groups[uf.find(i)].append(i)

    return [group for group in groups.values() if len(group) > 1]


def main() -> None:
    args = parse_args()
    image_paths = list_image_paths(args.image_dir, args.suffix)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {os.path.abspath(args.image_dir)}")

    print(f"Found {len(image_paths)} images")
    print("Extracting block-sum features...")
    coarse_feats, fine_feats, meta = extract_features(
        image_paths=image_paths,
        coarse_grid=tuple(args.coarse_grid),
        fine_grid=tuple(args.fine_grid),
    )

    raw_l1_threshold = args.threshold * 255.0 * meta["num_elements"]
    print(f"Image shape: {tuple(meta['image_shape'])}")
    print(f"Raw L1 threshold: {raw_l1_threshold:.3f}")

    print("Generating candidate pairs from coarse features...")
    coarse_pairs = candidate_pairs_from_coarse(coarse_feats, raw_l1_threshold)
    print(f"Candidates after coarse KD-tree: {len(coarse_pairs)}")

    print("Pruning candidates with fine features...")
    fine_pairs = prune_with_fine_feature(coarse_pairs, fine_feats, raw_l1_threshold)
    print(f"Candidates after fine lower bound: {len(fine_pairs)}")

    print("Running exact MAD verification...")
    verified_pairs = exact_pairs(image_paths, fine_pairs, raw_l1_threshold)
    print(f"Exact pairs found: {len(verified_pairs)}")

    components = connected_components(len(image_paths), verified_pairs)
    components_as_paths = [[image_paths[i] for i in group] for group in components]

    pair_records = [
        {
            "i": i,
            "j": j,
            "file_i": image_paths[i],
            "file_j": image_paths[j],
        }
        for i, j in verified_pairs
    ]

    result = {
        "image_dir": os.path.abspath(args.image_dir),
        "num_images": len(image_paths),
        "threshold": args.threshold,
        "criterion": "np.abs(x - y).mean() < threshold with x,y normalized to [0,1]",
        "warning": (
            "If your x and y are uint8 arrays, cast before subtraction. "
            "Use np.abs(x.astype(np.int16) - y.astype(np.int16)).mean() / 255."
        ),
        "image_shape": meta["image_shape"],
        "num_elements": meta["num_elements"],
        "raw_l1_threshold": raw_l1_threshold,
        "coarse_grid": list(args.coarse_grid),
        "fine_grid": list(args.fine_grid),
        "num_coarse_candidates": int(len(coarse_pairs)),
        "num_fine_candidates": int(len(fine_pairs)),
        "num_exact_pairs": int(len(verified_pairs)),
        "num_connected_components": int(len(components_as_paths)),
        "exact_pairs": pair_records,
        "connected_components": components_as_paths,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved result to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
