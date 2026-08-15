from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas import Candidate

FEATURE_NAMES = (
    "user_cf_score",
    "user_cf_available",
    "item_cf_score",
    "item_cf_available",
    "preference_match",
    "want_to_try",
    "group_preference_rate",
    "group_available",
    "context_match",
    "cleanliness_observed",
)


class ModelPackageError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ModelPackage:
    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    collaborative_index: dict[str, Any] | None

    @classmethod
    def load(cls, directory: Path) -> ModelPackage:
        manifest_path = directory / "manifest.json"
        try:
            manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
            _validate_manifest(manifest)
            artifact = directory / str(manifest["artifact"])
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != manifest["artifactSha256"]:
                raise ModelPackageError("model artifact checksum mismatch")
            with np.load(artifact, allow_pickle=False) as loaded:
                feature_names = tuple(str(value) for value in loaded["feature_names"].tolist())
                weights = np.asarray(loaded["weights"], dtype=float)
                mean = np.asarray(loaded["mean"], dtype=float)
                std = np.asarray(loaded["std"], dtype=float)
            collaborative_index = _load_collaborative_index(directory, manifest)
        except ModelPackageError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ModelPackageError("model package could not be loaded") from exc
        if feature_names != FEATURE_NAMES or tuple(manifest["featureNames"]) != FEATURE_NAMES:
            raise ModelPackageError("model feature schema mismatch")
        if weights.shape != (len(FEATURE_NAMES) + 1,) or mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape:
            raise ModelPackageError("model tensor shape mismatch")
        valid_tensors = np.isfinite(weights).all() and np.isfinite(mean).all() and np.isfinite(std).all()
        if not valid_tensors or (std <= 0).any():
            raise ModelPackageError("model tensor contains invalid values")
        return cls(weights=weights, mean=mean, std=std, collaborative_index=collaborative_index)

    def score(self, model_user_key: str, candidate: Candidate) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
        evidence = candidate.evidence
        user_cf, item_cf = self.collaborative_signals(model_user_key, candidate.model_meal_key)
        values = np.array(
            [
                user_cf["score"] or 0.0,
                float(user_cf["available"]),
                item_cf["score"] or 0.0,
                float(item_cf["available"]),
                evidence.preference_match,
                float(evidence.want_to_try),
                evidence.group_preference_rate or 0.0,
                float(evidence.group_eligible_member_count > 0),
                evidence.context_match or 0.0,
                float(evidence.cleanliness_observed),
            ],
            dtype=float,
        )
        standardised = (values - self.mean) / self.std
        score = float(self.weights[0] + standardised @ self.weights[1:])
        probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, score))))
        return probability, max(-100.0, min(100.0, score)), user_cf, item_cf

    def collaborative_signals(self, model_user_key: str, model_meal_key: str) -> tuple[dict[str, Any], dict[str, Any]]:
        unavailable_user = {"available": False, "score": None, "neighborSupport": 0}
        unavailable_item = {"available": False, "score": None, "supportingItemCount": 0}
        if self.collaborative_index is None:
            return unavailable_user, unavailable_item
        user = self.collaborative_index.get("userCf", {}).get(model_user_key, {}).get(model_meal_key)
        item = self.collaborative_index.get("itemCf", {}).get(model_user_key, {}).get(model_meal_key)
        user_signal = unavailable_user if user is None else {
            "available": True, "score": float(user["score"]), "neighborSupport": int(user["support"])
        }
        item_signal = unavailable_item if item is None else {
            "available": True, "score": float(item["score"]), "supportingItemCount": int(item["support"])
        }
        return user_signal, item_signal


def _validate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "packageVersion": "recommendation-package-v1",
        "modelVersion": "hybrid-ranking-v1",
        "featureSchemaVersion": "recommendation-features-v2",
        "inferenceContractVersion": "recommendation-inference-v1",
        "modelKeyVersion": "hmac-sha256-v1",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ModelPackageError("model package compatibility mismatch")
    if manifest.get("approvedFor") != ["local"]:
        raise ModelPackageError("model package is not approved for local runtime")
    if not isinstance(manifest.get("artifact"), str) or Path(str(manifest["artifact"])).name != manifest["artifact"]:
        raise ModelPackageError("model artifact path is invalid")
    if not isinstance(manifest.get("artifactSha256"), str) or len(manifest["artifactSha256"]) != 64:
        raise ModelPackageError("model artifact checksum is invalid")


def _load_collaborative_index(directory: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    metadata = manifest.get("collaborativeIndex")
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get("schemaVersion") != "foodmind-collaborative-index-v1":
        raise ModelPackageError("collaborative index compatibility mismatch")
    if metadata.get("positiveOnly") is not True:
        raise ModelPackageError("collaborative index must be positive-only")
    filename = metadata.get("artifact")
    checksum = metadata.get("artifactSha256")
    if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(checksum, str):
        raise ModelPackageError("collaborative index metadata is invalid")
    source_checksum = metadata.get("sourceSnapshotSha256")
    if not isinstance(source_checksum, str) or len(source_checksum) != 64:
        raise ModelPackageError("collaborative index source checksum is invalid")
    raw = (directory / filename).read_bytes()
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise ModelPackageError("collaborative index checksum mismatch")
    parsed = json.loads(raw)
    if (parsed.get("schemaVersion") != metadata["schemaVersion"]
            or parsed.get("positiveOnly") is not True
            or parsed.get("sourceSnapshotSha256") != source_checksum):
        raise ModelPackageError("collaborative index content is invalid")
    for section in ("userCf", "itemCf"):
        if not isinstance(parsed.get(section), dict):
            raise ModelPackageError("collaborative index content is invalid")
    return parsed
