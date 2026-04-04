"""Artifact packaging and management.

Handles packaging trained models for deployment.
"""

import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.shared_state import load_shared_state

logger = logging.getLogger("hand_sign_detection.training.artifact_manager")


class ArtifactManager:
    """Manager for model artifacts and deployment packages.

    Handles:
    - Packaging models for deployment
    - Creating checksums for validation
    - Managing artifact versions

    Usage:
        manager = ArtifactManager()
        package_path = manager.package_artifacts(profile_name="pi_zero")
    """

    def __init__(self):
        self._settings = get_settings()

    def package_artifacts(
        self,
        profile_name: str = "pi_zero",
        note: str = "",
        output_dir: str | None = None,
    ) -> str:
        """Package model artifacts for deployment.

        Creates a ZIP file containing:
        - RandomForest model and labels
        - LSTM model and labels (if available)
        - Shared state configuration
        - Checksums for validation

        Args:
            profile_name: Device profile name for the package
            note: Optional note to include in package
            output_dir: Output directory (uses models/packages if None)

        Returns:
            Path to created package ZIP file
        """
        settings = self._settings

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_id = f"local_{profile_name}_{timestamp}"

        if output_dir is None:
            output_dir = os.path.join(settings.models_dir, "packages")
        os.makedirs(output_dir, exist_ok=True)

        logger.info("Packaging artifacts: package_id=%s, profile=%s", package_id, profile_name)

        # Collect files to package
        files_to_copy = [
            os.path.join(settings.models_dir, "hand_alphabet_model.pkl"),
            os.path.join(settings.models_dir, "class_labels.npy"),
        ]

        # Add LSTM files if available
        lstm_model_path = os.path.join(settings.models_dir, "gesture_model.h5")
        lstm_labels_path = os.path.join(settings.models_dir, "wlasl_labels.npy")
        if os.path.exists(lstm_model_path):
            files_to_copy.append(lstm_model_path)
        if os.path.exists(lstm_labels_path):
            files_to_copy.append(lstm_labels_path)

        # Create temp directory for staging
        with tempfile.TemporaryDirectory() as staging_dir:
            # Copy files
            for file_path in files_to_copy:
                if os.path.exists(file_path):
                    dest_path = os.path.join(staging_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dest_path)
                    logger.debug("Copied %s", os.path.basename(file_path))

            # Create manifest
            manifest = self._create_manifest(
                staging_dir,
                package_id,
                profile_name,
                note,
            )

            manifest_path = os.path.join(staging_dir, "manifest.json")
            import json

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # Create ZIP
            zip_path = os.path.join(output_dir, f"{package_id}.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(staging_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, staging_dir)
                        zf.write(file_path, arcname)

            logger.info("Package created: %s", zip_path)
            return zip_path

    def _create_manifest(
        self,
        staging_dir: str,
        package_id: str,
        profile_name: str,
        note: str,
    ) -> dict:
        """Create package manifest with checksums."""
        settings = self._settings
        state = load_shared_state()

        files = []
        for filename in os.listdir(staging_dir):
            file_path = os.path.join(staging_dir, filename)
            if os.path.isfile(file_path):
                checksum = self._compute_checksum(file_path)
                files.append(
                    {
                        "name": filename,
                        "size": os.path.getsize(file_path),
                        "sha256": checksum,
                    }
                )

        return {
            "package_id": package_id,
            "profile": profile_name,
            "created_at": datetime.now().isoformat(),
            "note": note,
            "feature_schema": settings.feature_schema,
            "feature_schema_version": settings.feature_schema_version,
            "files": files,
            "shared_state": state,
        }

    def _compute_checksum(self, file_path: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_packages(self, output_dir: str | None = None) -> list[dict]:
        """List available packages.

        Args:
            output_dir: Directory to search (uses models/packages if None)

        Returns:
            List of package info dictionaries
        """
        settings = self._settings

        if output_dir is None:
            output_dir = os.path.join(settings.models_dir, "packages")

        if not os.path.exists(output_dir):
            return []

        packages = []
        for filename in os.listdir(output_dir):
            if filename.endswith(".zip"):
                file_path = os.path.join(output_dir, filename)
                packages.append(
                    {
                        "name": filename,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "created_at": datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat(),
                    }
                )

        return sorted(packages, key=lambda p: p["created_at"], reverse=True)


def package_artifacts(**kwargs) -> str:
    """Convenience function for packaging artifacts."""
    manager = ArtifactManager()
    return manager.package_artifacts(**kwargs)
