"""Unsupervised learning module for gesture discovery.

Provides:
- K-Means clustering for discovering gesture groups
- Autoencoder for feature learning and dimensionality reduction
- Cluster-based anomaly detection
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from hand_sign_detection.core.config import get_settings

logger = logging.getLogger("hand_sign_detection.training.unsupervised")


@dataclass
class ClusteringResult:
    """Result from clustering analysis."""

    n_clusters: int
    labels: np.ndarray
    cluster_centers: np.ndarray
    inertia: float
    silhouette_score: float | None = None
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def get_cluster_samples(self, cluster_id: int, X: np.ndarray) -> np.ndarray:
        """Get all samples belonging to a cluster."""
        mask = self.labels == cluster_id
        return X[mask]


@dataclass
class AutoencoderResult:
    """Result from autoencoder training."""

    encoding_dim: int
    reconstruction_loss: float
    encoded_data: np.ndarray
    reconstructed_data: np.ndarray | None = None


class GestureClusterer:
    """K-Means based gesture clustering for unsupervised learning.

    Discovers natural gesture groupings in unlabeled data.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 10,
    ):
        """Initialize the clusterer.

        Args:
            n_clusters: Number of clusters to discover
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for K-Means
            n_init: Number of times to run with different seeds
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self._model = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> ClusteringResult:
        """Fit the clusterer to data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            ClusteringResult with cluster assignments
        """
        from sklearn.cluster import KMeans

        logger.info("Fitting K-Means with %d clusters on %d samples", self.n_clusters, len(X))

        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )

        labels = self._model.fit_predict(X)
        self._fitted = True

        # Calculate silhouette score if we have enough clusters
        silhouette = None
        if self.n_clusters > 1 and len(np.unique(labels)) > 1:
            try:
                from sklearn.metrics import silhouette_score

                silhouette = silhouette_score(X, labels)
                logger.info("Silhouette score: %.4f", silhouette)
            except Exception as e:
                logger.warning("Could not compute silhouette score: %s", e)

        # Count cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

        return ClusteringResult(
            n_clusters=self.n_clusters,
            labels=labels,
            cluster_centers=self._model.cluster_centers_,
            inertia=self._model.inertia_,
            silhouette_score=silhouette,
            cluster_sizes=cluster_sizes,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data.

        Args:
            X: Feature matrix

        Returns:
            Cluster labels
        """
        if not self._fitted:
            raise RuntimeError("Clusterer must be fitted before prediction")
        return self._model.predict(X)

    def find_optimal_clusters(
        self,
        X: np.ndarray,
        min_clusters: int = 2,
        max_clusters: int = 20,
    ) -> tuple[int, list[float]]:
        """Find optimal number of clusters using elbow method.

        Args:
            X: Feature matrix
            min_clusters: Minimum clusters to try
            max_clusters: Maximum clusters to try

        Returns:
            Tuple of (optimal_k, inertia_values)
        """
        from sklearn.cluster import KMeans

        max_clusters = min(max_clusters, len(X) - 1)
        inertias = []

        logger.info("Finding optimal clusters (range %d-%d)", min_clusters, max_clusters)

        for k in range(min_clusters, max_clusters + 1):
            km = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=self.n_init,
            )
            km.fit(X)
            inertias.append(km.inertia_)

        # Find elbow using second derivative
        if len(inertias) >= 3:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_idx = np.argmax(diffs2) + 1  # +1 because of diff
            optimal_k = min_clusters + optimal_idx
        else:
            optimal_k = min_clusters

        logger.info("Optimal clusters: %d", optimal_k)
        return optimal_k, inertias

    def save(self, path: str | Path) -> None:
        """Save the fitted model."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model")
        joblib.dump(
            {
                "model": self._model,
                "n_clusters": self.n_clusters,
                "random_state": self.random_state,
            },
            path,
        )
        logger.info("Saved clusterer to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a saved model."""
        data = joblib.load(path)
        self._model = data["model"]
        self.n_clusters = data["n_clusters"]
        self.random_state = data["random_state"]
        self._fitted = True
        logger.info("Loaded clusterer from %s", path)


class FeatureAutoencoder:
    """Autoencoder for unsupervised feature learning.

    Uses a simple neural network to learn compressed representations
    of hand gesture features.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        encoding_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        """Initialize the autoencoder.

        Args:
            input_dim: Input feature dimension (auto-detected if None)
            encoding_dim: Size of the encoded representation
            hidden_dims: Hidden layer sizes (auto-configured if None)
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self._model = None
        self._encoder = None
        self._fitted = False

    def _build_model(self, input_dim: int):
        """Build the autoencoder architecture."""
        try:
            from tensorflow import keras
        except ImportError as err:
            raise ImportError(
                "TensorFlow required for autoencoder. Install with: pip install tensorflow"
            ) from err

        # Default hidden dimensions
        if self.hidden_dims is None:
            self.hidden_dims = [32, 16] if input_dim <= 64 else [64, 32]

        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for dim in self.hidden_dims:
            x = keras.layers.Dense(dim, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
        encoded = keras.layers.Dense(self.encoding_dim, activation="relu", name="encoding")(x)

        # Decoder (mirror of encoder)
        x = encoded
        for dim in reversed(self.hidden_dims):
            x = keras.layers.Dense(dim, activation="relu")(x)
            x = keras.layers.BatchNormalization()(x)
        decoded = keras.layers.Dense(input_dim, activation="linear")(x)

        # Full autoencoder
        self._model = keras.Model(inputs, decoded)
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
        )

        # Encoder-only model for feature extraction
        self._encoder = keras.Model(inputs, encoded)

        logger.info(
            "Built autoencoder: %d -> %s -> %d -> %s -> %d",
            input_dim,
            self.hidden_dims,
            self.encoding_dim,
            list(reversed(self.hidden_dims)),
            input_dim,
        )

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
        early_stopping_patience: int = 10,
    ) -> AutoencoderResult:
        """Train the autoencoder.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            early_stopping_patience: Patience for early stopping

        Returns:
            AutoencoderResult with encoded data
        """
        try:
            from tensorflow import keras
        except ImportError as err:
            raise ImportError("TensorFlow required for autoencoder") from err

        input_dim = X.shape[1]
        if self.input_dim is None:
            self.input_dim = input_dim

        self._build_model(input_dim)

        logger.info("Training autoencoder on %d samples for up to %d epochs", len(X), epochs)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        history = self._model.fit(
            X,
            X,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
        )

        self._fitted = True
        final_loss = history.history["val_loss"][-1]

        # Get encoded representations
        encoded_data = self._encoder.predict(X, verbose=0)

        logger.info(
            "Autoencoder trained. Final val_loss: %.6f, epochs: %d",
            final_loss,
            len(history.history["loss"]),
        )

        return AutoencoderResult(
            encoding_dim=self.encoding_dim,
            reconstruction_loss=final_loss,
            encoded_data=encoded_data,
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode features to lower-dimensional space.

        Args:
            X: Feature matrix

        Returns:
            Encoded features
        """
        if not self._fitted:
            raise RuntimeError("Autoencoder must be fitted before encoding")
        return self._encoder.predict(X, verbose=0)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct features (encode then decode).

        Args:
            X: Feature matrix

        Returns:
            Reconstructed features
        """
        if not self._fitted:
            raise RuntimeError("Autoencoder must be fitted before reconstruction")
        return self._model.predict(X, verbose=0)

    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction error.

        Useful for anomaly detection - high error indicates unusual samples.

        Args:
            X: Feature matrix

        Returns:
            Array of reconstruction errors (MSE per sample)
        """
        reconstructed = self.reconstruct(X)
        return np.mean((X - reconstructed) ** 2, axis=1)

    def save(self, path: str | Path) -> None:
        """Save the autoencoder model."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model")

        path = Path(path)
        self._model.save(path / "autoencoder.keras")
        self._encoder.save(path / "encoder.keras")

        # Save metadata
        joblib.dump(
            {
                "input_dim": self.input_dim,
                "encoding_dim": self.encoding_dim,
                "hidden_dims": self.hidden_dims,
            },
            path / "metadata.pkl",
        )

        logger.info("Saved autoencoder to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a saved autoencoder."""
        try:
            from tensorflow import keras
        except ImportError as err:
            raise ImportError("TensorFlow required to load autoencoder") from err

        path = Path(path)
        self._model = keras.models.load_model(path / "autoencoder.keras")
        self._encoder = keras.models.load_model(path / "encoder.keras")

        metadata = joblib.load(path / "metadata.pkl")
        self.input_dim = metadata["input_dim"]
        self.encoding_dim = metadata["encoding_dim"]
        self.hidden_dims = metadata["hidden_dims"]
        self._fitted = True

        logger.info("Loaded autoencoder from %s", path)


class UnsupervisedTrainer:
    """High-level trainer for unsupervised learning on gesture data.

    Combines clustering and autoencoder for comprehensive analysis.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        models_dir: str | None = None,
    ):
        """Initialize the trainer.

        Args:
            data_dir: Directory containing training data
            models_dir: Directory for saving models
        """
        settings = get_settings()
        self.data_dir = data_dir or settings.data_dir
        self.models_dir = models_dir or settings.models_dir

    def load_data(
        self,
        x_path: str | None = None,
    ) -> np.ndarray:
        """Load feature data from disk.

        Args:
            x_path: Path to X_data.npy (uses default if None)

        Returns:
            Feature matrix
        """
        if x_path is None:
            x_path = os.path.join(self.data_dir, "X_data.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(f"Data file not found: {x_path}")

        X = np.load(x_path)
        logger.info("Loaded data: shape=%s", X.shape)

        # If 3D (sequences), flatten to 2D for clustering
        if X.ndim == 3:
            n_samples, seq_len, n_features = X.shape
            X = X.reshape(n_samples, -1)
            logger.info("Flattened 3D data to shape=%s", X.shape)

        return X

    def train_clustering(
        self,
        X: np.ndarray | None = None,
        n_clusters: int | None = None,
        auto_select_k: bool = True,
        save_model: bool = True,
    ) -> ClusteringResult:
        """Train K-Means clustering on data.

        Args:
            X: Feature matrix (loads from disk if None)
            n_clusters: Number of clusters (auto-selected if None and auto_select_k)
            auto_select_k: Whether to auto-select optimal k
            save_model: Whether to save the trained model

        Returns:
            ClusteringResult
        """
        if X is None:
            X = self.load_data()

        clusterer = GestureClusterer(n_clusters=n_clusters or 10)

        # Auto-select k if requested
        if auto_select_k and n_clusters is None:
            optimal_k, _ = clusterer.find_optimal_clusters(X)
            clusterer.n_clusters = optimal_k

        result = clusterer.fit(X)

        if save_model:
            model_path = os.path.join(self.models_dir, "gesture_clusterer.pkl")
            clusterer.save(model_path)

        logger.info(
            "Clustering complete: %d clusters, inertia=%.2f, silhouette=%.4f",
            result.n_clusters,
            result.inertia,
            result.silhouette_score or 0,
        )

        return result

    def train_autoencoder(
        self,
        X: np.ndarray | None = None,
        encoding_dim: int = 16,
        epochs: int = 100,
        save_model: bool = True,
    ) -> AutoencoderResult:
        """Train autoencoder on data.

        Args:
            X: Feature matrix (loads from disk if None)
            encoding_dim: Dimension of encoded representation
            epochs: Training epochs
            save_model: Whether to save the trained model

        Returns:
            AutoencoderResult
        """
        if X is None:
            X = self.load_data()

        autoencoder = FeatureAutoencoder(encoding_dim=encoding_dim)
        result = autoencoder.fit(X, epochs=epochs)

        if save_model:
            model_dir = os.path.join(self.models_dir, "autoencoder")
            os.makedirs(model_dir, exist_ok=True)
            autoencoder.save(model_dir)

        logger.info(
            "Autoencoder trained: encoding_dim=%d, loss=%.6f",
            result.encoding_dim,
            result.reconstruction_loss,
        )

        return result

    def analyze_data(
        self,
        X: np.ndarray | None = None,
    ) -> dict:
        """Run comprehensive unsupervised analysis on data.

        Args:
            X: Feature matrix (loads from disk if None)

        Returns:
            Dictionary with analysis results
        """
        if X is None:
            X = self.load_data()

        results = {
            "n_samples": len(X),
            "n_features": X.shape[1] if X.ndim == 2 else X.shape[1] * X.shape[2],
        }

        # Clustering analysis
        logger.info("Running clustering analysis...")
        cluster_result = self.train_clustering(X, auto_select_k=True, save_model=True)
        results["clustering"] = {
            "n_clusters": cluster_result.n_clusters,
            "silhouette_score": cluster_result.silhouette_score,
            "cluster_sizes": cluster_result.cluster_sizes,
        }

        # Autoencoder analysis (if TensorFlow available)
        settings = get_settings()
        if settings.tensorflow_available:
            logger.info("Running autoencoder analysis...")
            try:
                ae_result = self.train_autoencoder(X, save_model=True)
                results["autoencoder"] = {
                    "encoding_dim": ae_result.encoding_dim,
                    "reconstruction_loss": ae_result.reconstruction_loss,
                }
            except Exception as e:
                logger.warning("Autoencoder analysis failed: %s", e)
                results["autoencoder"] = {"error": str(e)}
        else:
            results["autoencoder"] = {"error": "TensorFlow not available"}

        return results
