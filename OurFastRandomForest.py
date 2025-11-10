import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed

# -------------------------
# Implementação da Árvore Otimizada
# -------------------------
class OurFastDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples = X.shape[0]
        
        # Critérios de parada
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(np.unique(y)) == 1 or \
           num_samples < self.min_samples_split:
            return Counter(y).most_common(1)[0][0]

        # Seleção de características
        feat_idxs = np.random.choice(X.shape[1], self.n_features, replace=False)
        
        # Encontrar melhor split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            return Counter(y).most_common(1)[0][0]

        # Split dos dados
        mask = X[:, best_feat] <= best_thresh
        if np.sum(mask) == 0 or np.sum(~mask) == 0:
            return Counter(y).most_common(1)[0][0]

        # Crescimento recursivo
        left_child = self._grow_tree(X[mask], y[mask], depth + 1)
        right_child = self._grow_tree(X[~mask], y[~mask], depth + 1)
        
        return (best_feat, best_thresh, left_child, right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gini = 1.0
        best_idx, best_thresh = None, None

        for idx in feat_idxs:
            # Usar percentis para reduzir número de thresholds testados
            feature_values = X[:, idx]
            if len(np.unique(feature_values)) > 10:
                thresholds = np.percentile(feature_values, [25, 50, 75])
            else:
                thresholds = np.unique(feature_values)
            
            for thr in thresholds:
                gini = self._fast_gini_index(y, feature_values, thr)
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thresh = thr
        
        return best_idx, best_thresh

    def _fast_gini_index(self, y, feature_values, threshold):
        left_mask = feature_values <= threshold
        n_left = np.sum(left_mask)
        n_right = len(y) - n_left
        
        if n_left == 0 or n_right == 0:
            return 1.0

        # Cálculo otimizado do Gini usando bincount
        def fast_gini(labels):
            counts = np.bincount(labels)
            probs = counts / len(labels)
            return 1.0 - np.sum(probs ** 2)

        left_gini = fast_gini(y[left_mask])
        right_gini = fast_gini(y[~left_mask])
        
        total = len(y)
        return (n_left / total) * left_gini + (n_right / total) * right_gini

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
# Implementação do Random Forest Otimizado
# -------------------------
class OurFastRandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, 
                 max_features=None, n_jobs=-1, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.trees = []
        
        # Paralelização do treinamento das árvores
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_tree)(X, y, i) for i in range(self.n_trees)
        )

    def _build_tree(self, X, y, tree_idx):
        if self.random_state is not None:
            np.random.seed(self.random_state + tree_idx)
            
        tree = OurFastDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.max_features
        )
        
        # Bootstrap sample com índices
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[idxs], y[idxs]
        
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        if not self.trees:
            raise ValueError("O modelo não foi treinado ainda. Chame fit() primeiro.")
            
        # Coletar todas as predições de uma vez
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Votação majoritária otimizada
        final_preds = []
        for i in range(X.shape[0]):
            preds = all_preds[:, i]
            final_preds.append(Counter(preds).most_common(1)[0][0])
            
        return np.array(final_preds)

    def predict_proba(self, X):
        """Retorna probabilidades das classes"""
        if not self.trees:
            raise ValueError("O modelo não foi treinado ainda. Chame fit() primeiro.")
            
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        n_classes = len(np.unique(np.concatenate([tree.predict(X) for tree in self.trees])))
        
        probas = []
        for i in range(X.shape[0]):
            preds = all_preds[:, i]
            counts = np.bincount(preds, minlength=n_classes)
            probas.append(counts / len(preds))
            
        return np.array(probas)