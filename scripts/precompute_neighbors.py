"""
Precompute top-K nearest neighbors for all firms (full 768-dim cosine).

Generates a compact JSON file for the static GitHub Pages site.

Run on UCloud:
    python scripts/precompute_neighbors.py

Input (from TBIC-MA-NO project):
    data/processed/embeddings_normed.npy       (420K x 768 L2-normalized)
    data/processed/embeddings_orgnr_order.npy
    data/processed/semantic_map_payload.json.gz (for orgnr ordering)

Output:
    neighbors.json.gz  (in current working directory)
"""

from __future__ import annotations

import gzip
import json
import time

import numpy as np

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust TBIC_DATA to where the TBIC project's processed data lives
# ---------------------------------------------------------------------------
TBIC_DATA = Path("/work/TBIC-MA-NO/data/processed")

EMB_NORMED_PATH = TBIC_DATA / "embeddings_normed.npy"
EMB_ORGNR_PATH = TBIC_DATA / "embeddings_orgnr_order.npy"
PAYLOAD_PATH = TBIC_DATA / "semantic_map_payload.json.gz"
OUTPUT_PATH = Path("neighbors.json.gz")

K = 10
BATCH_SIZE = 200


def main() -> None:
    t0 = time.time()

    def progress(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # Load payload to get orgnr ordering (this is the frontend's row index)
    progress("Loading payload for orgnr ordering")
    payload = json.loads(gzip.decompress(Path(PAYLOAD_PATH).read_bytes()))
    payload_orgnrs = payload["orgnr"]
    n_payload = payload["n"]
    progress(f"Payload: {n_payload:,} firms")

    # Load normalized embeddings
    progress("Loading normalized embeddings")
    emb_normed = np.load(EMB_NORMED_PATH)
    emb_orgnrs = np.load(EMB_ORGNR_PATH, allow_pickle=True)
    n_emb = emb_normed.shape[0]
    progress(f"Embeddings: {n_emb:,} x {emb_normed.shape[1]}")

    # Build embedding orgnr -> row lookup
    emb_orgnr_to_row = {str(o): i for i, o in enumerate(emb_orgnrs)}

    # Build mapping: payload index -> embedding row
    progress("Mapping payload indices to embedding rows")
    payload_to_emb = np.zeros(n_payload, dtype=np.int64)
    valid_mask = np.zeros(n_payload, dtype=bool)
    for pi, orgnr in enumerate(payload_orgnrs):
        ei = emb_orgnr_to_row.get(orgnr)
        if ei is not None:
            payload_to_emb[pi] = ei
            valid_mask[pi] = True
    n_valid = int(valid_mask.sum())
    progress(f"Mapped: {n_valid:,}/{n_payload:,} firms")

    # Reindex embedding matrix to payload order
    progress("Reindexing embedding matrix to payload order")
    emb_payload_order = emb_normed[payload_to_emb].astype(np.float32)
    del emb_normed  # free ~1.3 GB

    # Preallocate output
    all_indices = np.zeros((n_payload, K), dtype=np.int32)
    all_sims = np.zeros((n_payload, K), dtype=np.float32)

    # Batch computation
    n_batches = (n_payload + BATCH_SIZE - 1) // BATCH_SIZE
    progress(f"Computing top-{K} neighbors in {n_batches:,} batches of {BATCH_SIZE}")

    for batch_i in range(n_batches):
        start = batch_i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_payload)
        batch_q = emb_payload_order[start:end]  # (batch, 768)

        # Cosine similarities: (batch, 768) @ (768, N) -> (batch, N)
        sims = batch_q @ emb_payload_order.T

        # Zero out self-similarity
        for j in range(end - start):
            sims[j, start + j] = -2.0

        # Top-K via argpartition
        for j in range(end - start):
            row_sims = sims[j]
            top_k = np.argpartition(row_sims, -K)[-K:]
            top_k = top_k[np.argsort(row_sims[top_k])[::-1]]
            all_indices[start + j] = top_k
            all_sims[start + j] = row_sims[top_k]

        if batch_i % 100 == 0 or batch_i == n_batches - 1:
            elapsed = time.time() - t0
            pct = 100 * (batch_i + 1) / n_batches
            eta = elapsed / (batch_i + 1) * (n_batches - batch_i - 1)
            progress(
                f"  Batch {batch_i + 1:,}/{n_batches:,} ({pct:.1f}%) "
                f"— ETA {eta:.0f}s"
            )

    del emb_payload_order

    # Serialize as compact JSON
    progress("Serializing to JSON")
    neighbor_data = []
    for i in range(n_payload):
        row = []
        for j in range(K):
            row.append(int(all_indices[i, j]))
            row.append(round(float(all_sims[i, j]), 4))
        neighbor_data.append(row)

    output = {"k": K, "n": n_payload, "data": neighbor_data}
    raw = json.dumps(output, separators=(",", ":"))
    progress(f"Raw JSON: {len(raw) / 1e6:.1f} MB")

    compressed = gzip.compress(raw.encode("utf-8"), compresslevel=6)
    OUTPUT_PATH.write_bytes(compressed)
    progress(f"Saved: {OUTPUT_PATH} ({len(compressed) / 1e6:.1f} MB)")

    elapsed = time.time() - t0
    progress(f"Total: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    progress("Done.")


if __name__ == "__main__":
    main()
