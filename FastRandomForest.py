import numpy as np
from math import sqrt
from joblib import Parallel, delayed

class FastDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None, n_classes=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_classes = n_classes  # se conhecido a priori
        self.tree = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(int)
        n_samples, n_features = X.shape
        if self.n_classes is None:
            self.n_classes = int(np.max(self.y) + 1)
        if self.max_features is None:
            self.max_features = n_features
        if self.max_features == 'sqrt':
            # padrão similar ao sklearn: sqrt(n_features) para classificação
            self.max_features = max(1, int(sqrt(n_features)))
        # inicia recursão passando índices (evita cópias)
        idxs = np.arange(n_samples, dtype=np.int32)
        self.tree = self._grow_tree(idxs, depth=0)
        # limpa referências grandes se quiser liberar memória: (opcional)
        # del self.X, self.y

    def _grow_tree(self, idxs, depth):
        y = self.y[idxs]
        # nó folha?
        if (self.max_depth is not None and depth >= self.max_depth) \
           or idxs.size < self.min_samples_split \
           or np.unique(y).size == 1:
            # retorna a classe majoritária
            return int(np.bincount(y, minlength=self.n_classes).argmax())

        # amostra características
        n_features = self.X.shape[1]
        feat_idxs = self.rng.choice(n_features, self.max_features, replace=False)

        best_feat, best_thr, best_score, best_left_idx, best_right_idx = self._best_split_vectorized(idxs, feat_idxs)
        if best_feat is None:
            return int(np.bincount(y, minlength=self.n_classes).argmax())

        # cria nó: (feat, thr, left_subtree, right_subtree)
        left = self._grow_tree(best_left_idx, depth + 1)
        right = self._grow_tree(best_right_idx, depth + 1)
        return (int(best_feat), float(best_thr), left, right)

    def _best_split_vectorized(self, idxs, feat_idxs):
        X = self.X
        y = self.y
        m = idxs.size
        if m <= 1:
            return (None, None, None, None, None)

        parent_counts = np.bincount(y[idxs], minlength=self.n_classes)
        parent_gini = 1.0 - np.sum((parent_counts / m) ** 2)

        best_gini = 1.0
        best_feat = None
        best_thr = None
        best_left_idx = None
        best_right_idx = None

        for feat in feat_idxs:
            vals = X[idxs, feat]
            order = np.argsort(vals, kind='mergesort')  # stable sort
            sorted_vals = vals[order]
            sorted_labels = y[idxs][order]

            # posições possíveis: onde valor muda
            diff_positions = np.where(sorted_vals[:-1] != sorted_vals[1:])[0]
            if diff_positions.size == 0:
                continue

            # Para cada classe, construir soma cumulativa (vetorizado por classe)
            # result: (n_classes, m)
            # Se n_classes pequeno, isto é eficiente.
            cum_counts = np.vstack([np.cumsum(sorted_labels == c) for c in range(self.n_classes)])  # shape: (C, m)

            # left_counts for candidate positions (index pos -> left includes indices 0..pos)
            left_counts_at_pos = cum_counts[:, diff_positions]  # shape: (C, n_pos)
            left_totals = np.sum(left_counts_at_pos, axis=0)
            right_counts_at_pos = parent_counts[:, None] - left_counts_at_pos
            right_totals = m - left_totals

            # calcular Gini left and right para cada pos:
            # gini_left = 1 - sum_i (left_counts_i / left_total)^2
            # evitar divide by zero pois diff_positions garante left_totals>0 e right_totals>0
            left_probs_sq = (left_counts_at_pos / left_totals) ** 2
            right_probs_sq = (right_counts_at_pos / right_totals) ** 2

            gini_left = 1.0 - np.sum(left_probs_sq, axis=0)
            gini_right = 1.0 - np.sum(right_probs_sq, axis=0)

            # weighted gini
            weighted_gini = (left_totals / m) * gini_left + (right_totals / m) * gini_right

            # encontra melhor posição nessa feature
            min_pos = np.argmin(weighted_gini)
            min_gini = weighted_gini[min_pos]
            if min_gini < best_gini:
                best_gini = float(min_gini)
                best_feat = feat
                pos = diff_positions[min_pos]
                # threshold entre sorted_vals[pos] e sorted_vals[pos+1] (média)
                best_thr = (sorted_vals[pos] + sorted_vals[pos + 1]) / 2.0
                # reconstrói índices para as duas partições (usando comparação sobre original indices)
                mask_left = X[idxs, feat] <= best_thr
                best_left_idx = idxs[mask_left]
                best_right_idx = idxs[~mask_left]

        return best_feat, best_thr, best_gini, best_left_idx, best_right_idx

    def predict(self, X):
        # prealoca
        preds = np.empty(X.shape[0], dtype=int)
        for i, row in enumerate(X):
            preds[i] = self._predict_one(row, self.tree)
        return preds

    def _predict_one(self, row, node):
        # se folha (int)
        if not isinstance(node, tuple):
            return int(node)
        feat, thr, left, right = node
        if row[feat] <= thr:
            return self._predict_one(row, left)
        else:
            return self._predict_one(row, right)


class FastRandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None,
                 n_jobs=1, random_state=None, n_classes=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.n_classes = n_classes
        self.trees = []

    def _build_one_tree(self, X, y, seed):
        n = len(X)
        # bootstrap via randint
        idxs = np.random.default_rng(seed).integers(0, n, n, dtype=np.int32)
        tree = FastDecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                max_features=self.max_features,
                                n_classes=self.n_classes,
                                random_state=seed)
        tree.fit(X[idxs], y[idxs])
        return tree

    def fit(self, X, y):
        seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_trees)
        # paraleliza construção (opcional)
        if self.n_jobs == 1:
            self.trees = [self._build_one_tree(X, y, int(seeds[i])) for i in range(self.n_trees)]
        else:
            self.trees = Parallel(n_jobs=self.n_jobs)(
                delayed(self._build_one_tree)(X, y, int(seeds[i])) for i in range(self.n_trees)
            )

    def predict(self, X):
        # coleta previsões (n_trees, n_samples)
        all_preds = np.vstack([t.predict(X) for t in self.trees])
        # votação (axis=0 -> por amostra)
        # usar bincount por coluna pode ser lento; fazemos por amostra
        n_samples = X.shape[0]
        final = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            col = all_preds[:, i]
            final[i] = int(np.bincount(col).argmax())
        return final