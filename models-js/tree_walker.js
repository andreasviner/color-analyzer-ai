// Auto-generated alongside *_trees.json by convert_to_js.py.
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
