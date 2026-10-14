"""Feature extraction for learned RIVER revision.

Standard RIVER-L features deliberately exclude transaction-time truth features.
``observed_at`` / ``transaction_time`` remain available in audit records but do
not enter the default feature vector.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from river.schemas import ValidityInterval


PROHIBITED_TRUTH_FEATURES = {
    "arrival_rank",
    "evidence_list_index",
    "turn_position",
    "last_mentioned_flag",
}


@dataclass(frozen=True)
class RiverLFeatureConfig:
    use_temporal_features: bool = True
    include_arrival_feature: bool = False
    max_interval_days: float = 3650.0

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return None


def _valid_time(value: Any) -> ValidityInterval:
    if isinstance(value, ValidityInterval):
        return value
    if value is None:
        return ValidityInterval()
    return ValidityInterval.model_validate(value)


class RiverLFeatureExtractor:
    """Converts assessed evidence into order-free per-evidence feature vectors."""

    BASE_FEATURE_NAMES = [
        "entailment_probability",
        "contradiction_probability",
        "neutral_probability",
        "temporal_applicability",
        "provenance_reliability",
        "extraction_confidence",
        "dependence_discount",
        "independence_group_fraction",
        "interval_uncertainty",
        "has_valid_start",
        "has_valid_end",
        "interval_width_fraction",
        "evidence_available",
        "evidence_completeness",
    ]

    def __init__(self, config: RiverLFeatureConfig | None = None):
        self.config = config or RiverLFeatureConfig()
        self.feature_names = list(self.BASE_FEATURE_NAMES)
        if self.config.include_arrival_feature:
            self.feature_names.append("arrival_recency_negative_control")

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def validate_standard_features(self) -> None:
        if PROHIBITED_TRUTH_FEATURES & set(self.feature_names):
            raise ValueError(f"prohibited truth-selection features present: {self.feature_names}")

    def evidence_features(
        self,
        evidence: dict[str, Any],
        *,
        group_counts: Counter[str],
        target_time: datetime | None,
        max_observed_at: datetime | None = None,
    ) -> list[float]:
        interval = _valid_time(evidence.get("valid_time"))
        group = str(evidence.get("independence_group", evidence.get("evidence_id", "")))
        group_count = max(group_counts.get(group, 1), 1)
        confidence = float(getattr(interval, "confidence", 1.0))
        start = interval.start
        end = interval.end

        if self.config.use_temporal_features:
            temporal = float(interval.contains(target_time))
            interval_uncertainty = max(0.0, 1.0 - confidence)
            has_start = 1.0 if start is not None else 0.0
            has_end = 1.0 if end is not None else 0.0
            if start is not None and end is not None:
                width_days = max((end - start).total_seconds() / 86400.0, 0.0)
                width_fraction = min(width_days / self.config.max_interval_days, 1.0)
            else:
                width_fraction = 1.0
        else:
            temporal = 1.0
            interval_uncertainty = 0.0
            has_start = 0.0
            has_end = 0.0
            width_fraction = 0.0

        text = str(evidence.get("text", ""))
        completeness_parts = [
            bool(text.strip()),
            bool(evidence.get("source_id")),
            bool(evidence.get("chunk_id")),
            bool(evidence.get("evidence_id")),
        ]
        features = [
            float(evidence.get("entailment", 0.0)),
            float(evidence.get("contradiction", 0.0)),
            float(evidence.get("neutral", 0.0)),
            temporal,
            float(evidence.get("provenance_reliability", 1.0)),
            float(evidence.get("extraction_confidence", 1.0)),
            1.0 / math.sqrt(group_count),
            min(group_count / 8.0, 1.0),
            interval_uncertainty,
            has_start,
            has_end,
            width_fraction,
            1.0,
            sum(completeness_parts) / len(completeness_parts),
        ]
        if self.config.include_arrival_feature:
            observed = _parse_datetime(evidence.get("observed_at") or evidence.get("transaction_time"))
            if observed is None or max_observed_at is None:
                recency = 0.0
            else:
                age_days = max((max_observed_at - observed).total_seconds() / 86400.0, 0.0)
                recency = math.exp(-0.01 * age_days)
            features.append(recency)
        return features


def target_time_from_example(example: dict[str, Any]) -> datetime | None:
    return _parse_datetime(example.get("target_time"))


def max_observed_time(evidence: list[dict[str, Any]]) -> datetime | None:
    values = [
        parsed
        for item in evidence
        if (parsed := _parse_datetime(item.get("observed_at") or item.get("transaction_time"))) is not None
    ]
    return max(values, default=None)
