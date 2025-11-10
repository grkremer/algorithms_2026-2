import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed

# -------------------------
# Implementação da Árvore
# -------------------------
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1] if not self.n_features else self.n_features
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]

        feat_idxs = np.random.choice(num_features, self.n_features, replace=False)

        # Melhor split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        # Splitar dados
        indices_left = X[:, best_feat] <= best_thresh
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return (best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gini = 1.0
        best_idx, best_thresh = None, None

        for idx in feat_idxs:
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                gini = self._gini_index(y, X[:, idx], thr)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = thr
        return best_idx, best_thresh

    def _gini_index(self, y, feature_values, threshold):
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 1.0

        def gini_group(labels):
            m = len(labels)
            counts = np.bincount(labels)
            return 1.0 - sum((count / m) ** 2 for count in counts if count > 0)

        left_gini = gini_group(y[left_mask])
        right_gini = gini_group(y[right_mask])
        m = len(y)
        return (np.sum(left_mask) / m) * left_gini + (np.sum(right_mask) / m) * right_gini

    def predict(self, X):
        return np.array([self._predict(inputs, self.tree) for inputs in X])

    def _predict(self, inputs, tree):
        if not isinstance(tree, tuple):
            return tree
        feat_idx, threshold, left_child, right_child = tree
        if inputs[feat_idx] <= threshold:
            return self._predict(inputs, left_child)
        else:
            return self._predict(inputs, right_child)

# -------------------------
# Implementação do Random Forest
# -------------------------
class OurRandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None, n_jobs=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.trees = []

    def _build_tree(self, X, y):
        idxs = np.random.choice(len(X), len(X), replace=True)
        X_sample, y_sample = X[idxs], y[idxs]
        
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.max_features
        )
        tree.fit(X_sample, y_sample)
        return tree

    def fit(self, X, y):
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y) for _ in range(self.n_trees)
        )

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Votação majoritária
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([Counter(preds).most_common(1)[0][0] for preds in tree_preds])