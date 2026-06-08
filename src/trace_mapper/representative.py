from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import pandas as pd


REQUIRED_COLUMNS = {
    "source_duration_s",
    "source_interarrival_s",
    "source_num_gpus",
}

RUNTIME_THRESHOLDS_S = {
    "under_10m": 10 * 60,
    "under_1h": 60 * 60,
    "under_6h": 6 * 60 * 60,
    "under_1d": 24 * 60 * 60,
}

DEFAULT_SCORE_WEIGHTS = {
    "gpu_demand": 0.25,
    "runtime_cdf": 0.35,
    "interarrival": 0.20,
    "burst_fraction": 0.05,
    "gpu_service": 0.15,
}


def _validate_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(jobs.columns)

    if missing:
        raise ValueError(
            "Jobs are missing required representative-profile "
            f"columns: {sorted(missing)}"
        )

    if jobs.empty:
        raise ValueError("Jobs must not be empty")

    frame = jobs.copy()

    frame["source_duration_s"] = pd.to_numeric(
        frame["source_duration_s"],
        errors="raise",
    )
    frame["source_interarrival_s"] = pd.to_numeric(
        frame["source_interarrival_s"],
        errors="raise",
    )
    frame["source_num_gpus"] = pd.to_numeric(
        frame["source_num_gpus"],
        errors="raise",
    ).astype(int)

    if (frame["source_duration_s"] <= 0).any():
        raise ValueError("source_duration_s must be positive")

    if (frame["source_interarrival_s"] < 0).any():
        raise ValueError(
            "source_interarrival_s must be nonnegative"
        )

    if (frame["source_num_gpus"] <= 0).any():
        raise ValueError("source_num_gpus must be positive")

    return frame


def build_representativeness_profile(
    jobs: pd.DataFrame,
    *,
    supported_gpu_counts: Iterable[int] = (1, 2),
) -> dict:
    frame = _validate_jobs(jobs)

    supported_counts = tuple(
        sorted({int(value) for value in supported_gpu_counts})
    )

    if not supported_counts:
        raise ValueError(
            "supported_gpu_counts must not be empty"
        )

    if any(value <= 0 for value in supported_counts):
        raise ValueError(
            "supported_gpu_counts must contain positive integers"
        )

    frame = frame.loc[
        frame["source_num_gpus"].isin(supported_counts)
    ].copy()

    if frame.empty:
        raise ValueError(
            "No jobs remain after GPU-demand filtering"
        )

    job_count = len(frame)

    gpu_demand_fractions = {
        str(gpu_count): float(
            (frame["source_num_gpus"] == gpu_count).mean()
        )
        for gpu_count in supported_counts
    }

    runtime_cdf = {
        label: float(
            (
                frame["source_duration_s"]
                < threshold_s
            ).mean()
        )
        for label, threshold_s
        in RUNTIME_THRESHOLDS_S.items()
    }

    interarrivals = frame["source_interarrival_s"]

    gpu_service_s = (
        frame["source_num_gpus"]
        * frame["source_duration_s"]
    )

    total_gpu_service_s = float(gpu_service_s.sum())

    gpu_service_fractions = {}

    for gpu_count in supported_counts:
        service = float(
            gpu_service_s.loc[
                frame["source_num_gpus"] == gpu_count
            ].sum()
        )

        gpu_service_fractions[str(gpu_count)] = (
            service / total_gpu_service_s
            if total_gpu_service_s > 0
            else 0.0
        )

    return {
        "job_count": int(job_count),
        "supported_gpu_counts": list(supported_counts),
        "gpu_demand_fractions": gpu_demand_fractions,
        "runtime_cdf": runtime_cdf,
        "interarrival_p50_s": float(
            interarrivals.quantile(0.50)
        ),
        "interarrival_p95_s": float(
            interarrivals.quantile(0.95)
        ),
        "zero_interarrival_fraction": float(
            (interarrivals == 0).mean()
        ),
        "duration_p50_s": float(
            frame["source_duration_s"].quantile(0.50)
        ),
        "duration_p95_s": float(
            frame["source_duration_s"].quantile(0.95)
        ),
        "gpu_service_fractions": gpu_service_fractions,
        "total_gpu_service_s": total_gpu_service_s,
    }


def _distribution_l1(
    candidate: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    keys = sorted(set(candidate) | set(target))

    return 0.5 * sum(
        abs(
            float(candidate.get(key, 0.0))
            - float(target.get(key, 0.0))
        )
        for key in keys
    )


def _mean_absolute_difference(
    candidate: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    keys = sorted(set(candidate) | set(target))

    if not keys:
        return 0.0

    return sum(
        abs(
            float(candidate.get(key, 0.0))
            - float(target.get(key, 0.0))
        )
        for key in keys
    ) / len(keys)


def _log_ratio_distance(
    candidate: float,
    target: float,
) -> float:
    # A value of 1.0 means approximately a factor-of-two
    # difference after adding one second to handle zeros.
    return abs(
        math.log2(
            (float(candidate) + 1.0)
            / (float(target) + 1.0)
        )
    )


def score_representativeness(
    candidate_profile: Mapping,
    target_profile: Mapping,
    *,
    weights: Mapping[str, float] | None = None,
) -> dict:
    score_weights = dict(
        DEFAULT_SCORE_WEIGHTS
        if weights is None
        else weights
    )

    missing_weights = (
        set(DEFAULT_SCORE_WEIGHTS)
        - set(score_weights)
    )

    if missing_weights:
        raise ValueError(
            "Missing representativeness score weights: "
            f"{sorted(missing_weights)}"
        )

    if any(float(value) < 0 for value in score_weights.values()):
        raise ValueError(
            "Representativeness score weights must be nonnegative"
        )

    weight_sum = sum(float(value) for value in score_weights.values())

    if weight_sum <= 0:
        raise ValueError(
            "Representativeness score weights must sum to "
            "a positive value"
        )

    normalized_weights = {
        key: float(value) / weight_sum
        for key, value in score_weights.items()
    }

    gpu_demand_distance = _distribution_l1(
        candidate_profile["gpu_demand_fractions"],
        target_profile["gpu_demand_fractions"],
    )

    runtime_cdf_distance = _mean_absolute_difference(
        candidate_profile["runtime_cdf"],
        target_profile["runtime_cdf"],
    )

    interarrival_distance = 0.5 * (
        _log_ratio_distance(
            candidate_profile["interarrival_p50_s"],
            target_profile["interarrival_p50_s"],
        )
        + _log_ratio_distance(
            candidate_profile["interarrival_p95_s"],
            target_profile["interarrival_p95_s"],
        )
    )

    burst_fraction_distance = abs(
        float(
            candidate_profile[
                "zero_interarrival_fraction"
            ]
        )
        - float(
            target_profile[
                "zero_interarrival_fraction"
            ]
        )
    )

    gpu_service_distance = _distribution_l1(
        candidate_profile["gpu_service_fractions"],
        target_profile["gpu_service_fractions"],
    )

    components = {
        "gpu_demand_distance": gpu_demand_distance,
        "runtime_cdf_distance": runtime_cdf_distance,
        "interarrival_distance": interarrival_distance,
        "burst_fraction_distance": burst_fraction_distance,
        "gpu_service_distance": gpu_service_distance,
    }

    total_score = (
        normalized_weights["gpu_demand"]
        * gpu_demand_distance
        + normalized_weights["runtime_cdf"]
        * runtime_cdf_distance
        + normalized_weights["interarrival"]
        * interarrival_distance
        + normalized_weights["burst_fraction"]
        * burst_fraction_distance
        + normalized_weights["gpu_service"]
        * gpu_service_distance
    )

    return {
        "total_score": float(total_score),
        "components": components,
        "weights": normalized_weights,
    }
