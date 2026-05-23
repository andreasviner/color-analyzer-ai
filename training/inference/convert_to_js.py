"""
Translate the full-size LightGBM boosters trained by train_final.py into
compact JSON files plus a tiny JavaScript tree walker, so the browser can run
the same model the leaderboard reports without downloading lightgbm.

Each *_trees.json file follows this schema:
    {
      "objective": "binary" | "regression",
      "n_features": 477 | 475,
      "trees": [
        # one flat list per tree, 4 entries per node:
        #   (feature_idx, threshold, left_child_idx, right_child_idx)  internal
        #   (-1,          leaf_value, 0, 0)                            leaf
        [0, 1.23, 1, 2, -1, 0.5, 0, 0, ...],
        ...
      ]
    }

The walker in `tree_walker.js` is ~30 lines and reused for all three models.

Outputs go to ../../models-js/ so survey.html can fetch them from a relative
path without needing the Cloudflare worker.
"""

import json
import os
import time

import numpy as np
import lightgbm as lgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "models-js"))
os.makedirs(OUT_DIR, exist_ok=True)


def _flatten_tree(node):
    """Iteratively flatten a LightGBM tree_structure into a flat node list.

    Each node occupies 4 consecutive entries in the returned list:
        internal: (feature_idx, threshold, left_idx, right_idx)
        leaf:     (-1, leaf_value, 0, 0)
    Returns the flat list (length = 4 * num_nodes)."""
    # First pass: build the tree as a list of "slot" tuples using BFS-ish
    # iteration. Pre-allocate a placeholder, then push children with their slot.
    nodes_flat = [None, None, None, None]  # placeholder for root
    stack = [(node, 0)]  # (subtree, slot index)
    while stack:
        n, slot = stack.pop()
        base = slot * 4
        if "leaf_index" in n:
            nodes_flat[base]     = -1
            nodes_flat[base + 1] = float(n["leaf_value"])
            nodes_flat[base + 2] = 0
            nodes_flat[base + 3] = 0
            continue
        # internal node - reserve two child slots
        left_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        right_idx = len(nodes_flat) // 4
        nodes_flat.extend([None, None, None, None])
        nodes_flat[base]     = int(n["split_feature"])
        nodes_flat[base + 1] = float(n["threshold"])
        nodes_flat[base + 2] = left_idx
        nodes_flat[base + 3] = right_idx
        # Push children. Visit left first by pushing right first (LIFO stack).
        stack.append((n["right_child"], right_idx))
        stack.append((n["left_child"], left_idx))
    return nodes_flat


def _emit_model(booster: lgb.Booster, out_path: str, objective: str):
    dump = booster.dump_model()
    n_features = int(dump["max_feature_idx"]) + 1
    trees = []
    total_nodes = 0
    for tree_info in dump["tree_info"]:
        flat = _flatten_tree(tree_info["tree_structure"])
        trees.append(flat)
        total_nodes += len(flat) // 4

    payload = {
        "objective": objective,
        "n_features": n_features,
        "n_trees": len(trees),
        "n_nodes": total_nodes,
        "trees": trees,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return total_nodes, len(trees)


def _verify(out_path: str, booster: lgb.Booster, X_sample: np.ndarray):
    """JSON-load, walk in pure Python the same way the JS walker will, and
    compare against booster.predict raw scores."""
    with open(out_path, encoding="utf-8") as fh:
        model = json.load(fh)

    def py_score(features):
        total = 0.0
        for tree in model["trees"]:
            i = 0
            while tree[i * 4] != -1:
                feat = tree[i * 4]
                thr = tree[i * 4 + 1]
                i = tree[i * 4 + 2] if features[feat] <= thr else tree[i * 4 + 3]
            total += tree[i * 4 + 1]
        return total

    raw_lgb = booster.predict(X_sample, raw_score=True)
    js_like = np.array([py_score(list(map(float, row))) for row in X_sample], dtype=np.float64)
    return float(np.max(np.abs(raw_lgb - js_like)))


TREE_WALKER_JS = """// Auto-generated alongside *_trees.json by convert_to_js.py.
// Loads a JSON model and exposes a `score(features)` function that walks the
// same flat-tree representation the Python converter wrote.
//
// Usage:
//   const model = await TreeWalker.load('/ai/color-polygraph/models-js/gender_trees.json');
//   const raw = TreeWalker.score(model, features);            // raw logit / regression value
//   const prob = TreeWalker.sigmoid(raw);                     // for binary classifiers

(function (global) {
  async function load(url) {
    const r = await fetch(url, { cache: 'force-cache' });
    if (!r.ok) throw new Error('Failed to load model: ' + url + ' (' + r.status + ')');
    return await r.json();
  }

  function score(model, features) {
    let total = 0;
    const trees = model.trees;
    for (let t = 0; t < trees.length; t++) {
      const tree = trees[t];
      let i = 0;
      while (tree[i * 4] !== -1) {
        const feat = tree[i * 4];
        const thr = tree[i * 4 + 1];
        i = (features[feat] <= thr) ? tree[i * 4 + 2] : tree[i * 4 + 3];
      }
      total += tree[i * 4 + 1];
    }
    return total;
  }

  function sigmoid(x) {
    if (x >= 0) {
      const e = Math.exp(-x);
      return 1 / (1 + e);
    } else {
      const e = Math.exp(x);
      return e / (1 + e);
    }
  }

  global.TreeWalker = { load, score, sigmoid };
})(typeof window !== 'undefined' ? window : globalThis);
"""


def main():
    t_total = time.time()

    targets = [
        ("gender", "gender_model.txt", "binary",     "gender_trees.json"),
        ("age",    "age_model.txt",    "regression", "age_trees.json"),
        ("mood",   "mood_model.txt",   "regression", "mood_trees.json"),
    ]

    # Quick sanity sample to verify each emitted file matches LightGBM.
    # Pull a few rows of the right shape from features.npy.
    X_base = np.load(os.path.join(HERE, "..", "features.npy"))
    X_extra = np.load(os.path.join(HERE, "..", "extra-features", "features_extra.npy"))
    X_static = np.concatenate([X_base, X_extra], axis=1).astype(np.float32)

    # The buckets script appends 3 (gender) or 1 (age/mood) extra feature(s).
    # For verification we don't need the actual bucket values - any finite floats
    # placed at the booster's expected positions will do, because we compare
    # booster.predict against our own walker on identical inputs.
    bucket_grid_path = os.path.join(HERE, "..", "color-buckets", "bucket_scores.npz")
    buckets = np.load(bucket_grid_path)
    girly = buckets["girly_grid"].reshape(512)
    masc = buckets["masc_grid"].reshape(512)
    age_grid = buckets["age_grid"].reshape(512)
    mood_grid = buckets["mood_grid"].reshape(512)

    # Use zeros for the smooth lookup so we get a deterministic verification
    # input. Just adds 0 (or 0-vector @ grid = 0) bucket columns. Inputs to the
    # actual model at runtime will obviously not be zero, but for the bit-exact
    # check it doesn't matter.
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_static.shape[0], 16, replace=False)
    X_sample_static = X_static[sample_idx]

    for target_name, booster_file, objective, json_name in targets:
        booster_path = os.path.join(HERE, booster_file)
        out_path = os.path.join(OUT_DIR, json_name)

        booster = lgb.Booster(model_file=booster_path)
        n_features = booster.num_feature()

        # Build matching X_sample with zero bucket columns appended
        n_extra = n_features - X_sample_static.shape[1]
        X_sample = np.concatenate(
            [X_sample_static, np.zeros((X_sample_static.shape[0], n_extra), dtype=np.float32)],
            axis=1,
        )

        print(f"Emitting {target_name}: {booster_file} -> {json_name}")
        t0 = time.time()
        n_nodes, n_trees = _emit_model(booster, out_path, objective)
        elapsed_emit = time.time() - t0

        t0 = time.time()
        delta = _verify(out_path, booster, X_sample)
        elapsed_verify = time.time() - t0

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  trees={n_trees}  nodes={n_nodes}  "
              f"json={size_mb:.2f} MB  max|delta|={delta:.2e}  "
              f"(emit {elapsed_emit:.1f}s, verify {elapsed_verify:.1f}s)")
        if delta > 1e-5:
            raise SystemExit(f"FAILED: {target_name} disagrees with LightGBM by {delta:.2e}")

    walker_path = os.path.join(OUT_DIR, "tree_walker.js")
    with open(walker_path, "w", encoding="utf-8") as fh:
        fh.write(TREE_WALKER_JS)
    print(f"Wrote {walker_path}  ({os.path.getsize(walker_path)} bytes)")

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
