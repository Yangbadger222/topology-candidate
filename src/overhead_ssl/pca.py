import numpy as np
from sklearn.decomposition import PCA


class SharedPCA:
    def __init__(self, n_components: int = 3, l2_normalize: bool = False):
        self.pca = PCA(n_components=n_components)
        self.l2_normalize = l2_normalize

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.l2_normalize:
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def fit(self, tokens: np.ndarray) -> "SharedPCA":
        self.pca.fit(self._prep(tokens))
        return self

    def transform_grid(self, features: np.ndarray) -> np.ndarray:
        h, w, c = features.shape
        z = self.pca.transform(self._prep(features.reshape(-1, c))).reshape(h, w, 3)
        z -= z.min(axis=(0, 1), keepdims=True)
        z /= np.maximum(z.max(axis=(0, 1), keepdims=True), 1e-8)
        return (z * 255).astype(np.uint8)

