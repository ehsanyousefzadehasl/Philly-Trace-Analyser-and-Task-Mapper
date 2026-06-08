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

SELECTION_REQUIRED_COLUMNS = {
    "source_job_id",
    "source_duration_s",
    "source_num_gpus",
    "submit_time_s",
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

def select_representative_job_window(
    source_jobs: pd.DataFrame,
    *,
    num_jobs: int,
    supported_gpu_counts: Iterable[int] = (1, 2),
    minimum_jobs_by_gpu_count: Mapping[int, int] | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    missing = SELECTION_REQUIRED_COLUMNS - set(
        source_jobs.columns
    )

    if missing:
        raise ValueError(
            "source_jobs is missing required representative "
            f"selection columns: {sorted(missing)}"
        )

    if num_jobs <= 0:
        raise ValueError("num_jobs must be positive")

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

    minimum_counts = {
        int(gpu_count): int(minimum)
        for gpu_count, minimum in (
            minimum_jobs_by_gpu_count or {}
        ).items()
        if int(minimum) > 0
    }

    for gpu_count, minimum in minimum_counts.items():
        if gpu_count not in supported_counts:
            raise ValueError(
                "minimum_jobs_by_gpu_count contains an "
                f"unsupported GPU count: {gpu_count}"
            )

        if minimum < 0:
            raise ValueError(
                "minimum_jobs_by_gpu_count values must be "
                "nonnegative"
            )

    if sum(minimum_counts.values()) > num_jobs:
        raise ValueError(
            "The sum of minimum_jobs_by_gpu_count exceeds "
            "num_jobs"
        )

    jobs = source_jobs.copy()

    jobs["source_job_id"] = jobs[
        "source_job_id"
    ].astype(str)

    jobs["source_duration_s"] = pd.to_numeric(
        jobs["source_duration_s"],
        errors="raise",
    )

    jobs["source_num_gpus"] = pd.to_numeric(
        jobs["source_num_gpus"],
        errors="raise",
    ).astype(int)

    jobs["submit_time_s"] = pd.to_numeric(
        jobs["submit_time_s"],
        errors="raise",
    )

    if (jobs["source_duration_s"] <= 0).any():
        raise ValueError(
            "source_duration_s must be positive"
        )

    if (jobs["source_num_gpus"] <= 0).any():
        raise ValueError(
            "source_num_gpus must be positive"
        )

    jobs = jobs.sort_values(
        ["submit_time_s", "source_job_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    eligible = jobs.loc[
        jobs["source_num_gpus"].isin(supported_counts)
    ].copy()

    if len(eligible) < num_jobs:
        raise ValueError(
            f"Requested {num_jobs} jobs, but only "
            f"{len(eligible)} eligible jobs are available"
        )

    # Recompute interarrival times after filtering unsupported jobs.
    eligible["source_interarrival_s"] = (
        eligible["submit_time_s"]
        .diff()
        .fillna(0.0)
    )

    target_profile = build_representativeness_profile(
        eligible,
        supported_gpu_counts=supported_counts,
    )

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

    weight_sum = sum(
        float(value)
        for value in score_weights.values()
    )

    if weight_sum <= 0:
        raise ValueError(
            "Representativeness score weights must sum "
            "to a positive value"
        )

    normalized_weights = {
        key: float(value) / weight_sum
        for key, value in score_weights.items()
    }

    window_size = int(num_jobs)
    rolling = eligible.rolling(
        window=window_size,
        min_periods=window_size,
    )

    metric_frame = pd.DataFrame(
        index=eligible.index
    )

    # GPU-demand fractions and minimum-count constraints.
    valid_window = pd.Series(
        True,
        index=eligible.index,
        dtype=bool,
    )

    gpu_demand_distance = pd.Series(
        0.0,
        index=eligible.index,
    )

    rolling_service_by_gpu: dict[int, pd.Series] = {}

    total_service = (
        eligible["source_num_gpus"]
        * eligible["source_duration_s"]
    )

    rolling_total_service = total_service.rolling(
        window_size,
        min_periods=window_size,
    ).sum()

    for gpu_count in supported_counts:
        indicator = (
            eligible["source_num_gpus"] == gpu_count
        ).astype(float)

        rolling_count = indicator.rolling(
            window_size,
            min_periods=window_size,
        ).sum()

        rolling_fraction = (
            rolling_count / window_size
        )

        target_fraction = float(
            target_profile[
                "gpu_demand_fractions"
            ][str(gpu_count)]
        )

        gpu_demand_distance += (
            rolling_fraction - target_fraction
        ).abs()

        required_minimum = minimum_counts.get(
            gpu_count,
            0,
        )

        if required_minimum > 0:
            valid_window &= (
                rolling_count >= required_minimum
            )

        service = total_service.where(
            eligible["source_num_gpus"] == gpu_count,
            0.0,
        )

        rolling_service_by_gpu[gpu_count] = (
            service.rolling(
                window_size,
                min_periods=window_size,
            ).sum()
        )

    gpu_demand_distance *= 0.5

    # Runtime-CDF distance.
    runtime_cdf_distance = pd.Series(
        0.0,
        index=eligible.index,
    )

    for label, threshold_s in (
        RUNTIME_THRESHOLDS_S.items()
    ):
        rolling_fraction = (
            (
                eligible["source_duration_s"]
                < threshold_s
            )
            .astype(float)
            .rolling(
                window_size,
                min_periods=window_size,
            )
            .mean()
        )

        target_fraction = float(
            target_profile["runtime_cdf"][label]
        )

        runtime_cdf_distance += (
            rolling_fraction - target_fraction
        ).abs()

    runtime_cdf_distance /= len(
        RUNTIME_THRESHOLDS_S
    )

    # Interarrival behavior.
    interarrivals = eligible[
        "source_interarrival_s"
    ]

    rolling_interarrival_p50 = (
        interarrivals.rolling(
            window_size,
            min_periods=window_size,
        ).quantile(0.50)
    )

    rolling_interarrival_p95 = (
        interarrivals.rolling(
            window_size,
            min_periods=window_size,
        ).quantile(0.95)
    )

    target_interarrival_p50 = float(
        target_profile["interarrival_p50_s"]
    )
    target_interarrival_p95 = float(
        target_profile["interarrival_p95_s"]
    )

    interarrival_distance = 0.5 * (
        (
            (
                rolling_interarrival_p50 + 1.0
            )
            / (
                target_interarrival_p50 + 1.0
            )
        )
        .map(math.log2)
        .abs()
        + (
            (
                rolling_interarrival_p95 + 1.0
            )
            / (
                target_interarrival_p95 + 1.0
            )
        )
        .map(math.log2)
        .abs()
    )

    rolling_burst_fraction = (
        (interarrivals == 0)
        .astype(float)
        .rolling(
            window_size,
            min_periods=window_size,
        )
        .mean()
    )

    burst_fraction_distance = (
        rolling_burst_fraction
        - float(
            target_profile[
                "zero_interarrival_fraction"
            ]
        )
    ).abs()

    # GPU-service distribution distance.
    gpu_service_distance = pd.Series(
        0.0,
        index=eligible.index,
    )

    for gpu_count in supported_counts:
        candidate_fraction = (
            rolling_service_by_gpu[gpu_count]
            / rolling_total_service
        )

        target_fraction = float(
            target_profile[
                "gpu_service_fractions"
            ][str(gpu_count)]
        )

        gpu_service_distance += (
            candidate_fraction - target_fraction
        ).abs()

    gpu_service_distance *= 0.5

    metric_frame["gpu_demand_distance"] = (
        gpu_demand_distance
    )
    metric_frame["runtime_cdf_distance"] = (
        runtime_cdf_distance
    )
    metric_frame["interarrival_distance"] = (
        interarrival_distance
    )
    metric_frame["burst_fraction_distance"] = (
        burst_fraction_distance
    )
    metric_frame["gpu_service_distance"] = (
        gpu_service_distance
    )

    metric_frame["total_score"] = (
        normalized_weights["gpu_demand"]
        * metric_frame["gpu_demand_distance"]
        + normalized_weights["runtime_cdf"]
        * metric_frame["runtime_cdf_distance"]
        + normalized_weights["interarrival"]
        * metric_frame["interarrival_distance"]
        + normalized_weights["burst_fraction"]
        * metric_frame["burst_fraction_distance"]
        + normalized_weights["gpu_service"]
        * metric_frame["gpu_service_distance"]
    )

    # The rolling result at row i represents the window
    # [i - window_size + 1, i].
    valid_window &= (
        eligible.index >= window_size - 1
    )
    valid_window &= metric_frame[
        "total_score"
    ].notna()

    valid_end_indices = metric_frame.index[
        valid_window
    ]

    if len(valid_end_indices) == 0:
        raise ValueError(
            "No representative window satisfies "
            f"minimum_jobs_by_gpu_count={minimum_counts}"
        )

    best_end_index = int(
        metric_frame.loc[
            valid_end_indices,
            "total_score",
        ].idxmin()
    )

    best_start_index = (
        best_end_index - window_size + 1
    )

    window = eligible.iloc[
        best_start_index:best_end_index + 1
    ].copy()

    window["source_trace_submit_time_s"] = (
        window["submit_time_s"]
    )
    window["source_trace_interarrival_s"] = (
        window["source_interarrival_s"]
    )

    first_submit_time = float(
        window["submit_time_s"].iloc[0]
    )

    window["submit_time_s"] = (
        window["submit_time_s"]
        - first_submit_time
    )

    window["source_interarrival_s"] = (
        window["submit_time_s"]
        .diff()
        .fillna(0.0)
    )

    window["source_window_job_index"] = range(
        len(window)
    )

    window = window.reset_index(drop=True)

    candidate_profile = (
        build_representativeness_profile(
            window,
            supported_gpu_counts=supported_counts,
        )
    )

    best_metrics = metric_frame.loc[
        best_end_index
    ]

    selected_gpu_counts = {
        str(int(gpu_count)): int(count)
        for gpu_count, count in (
            window["source_num_gpus"]
            .value_counts()
            .sort_index()
            .items()
        )
    }

    metadata = {
        "selection_method": (
            "best_representative_contiguous_eligible_window"
        ),
        "requested_job_count": int(num_jobs),
        "source_job_count": int(len(jobs)),
        "eligible_job_count": int(len(eligible)),
        "supported_gpu_counts": list(
            supported_counts
        ),
        "minimum_jobs_by_gpu_count": {
            str(key): int(value)
            for key, value in sorted(
                minimum_counts.items()
            )
        },
        "selected_jobs_by_gpu_count": (
            selected_gpu_counts
        ),
        "candidate_window_count": int(
            len(valid_end_indices)
        ),
        "eligible_start_index": int(
            best_start_index
        ),
        "eligible_end_index_exclusive": int(
            best_end_index + 1
        ),
        "first_source_job_id": str(
            window["source_job_id"].iloc[0]
        ),
        "last_source_job_id": str(
            window["source_job_id"].iloc[-1]
        ),
        "original_first_submit_time_s": (
            first_submit_time
        ),
        "original_last_submit_time_s": float(
            window[
                "source_trace_submit_time_s"
            ].iloc[-1]
        ),
        "selected_arrival_span_s": float(
            window["submit_time_s"].iloc[-1]
        ),
        "representativeness": {
            "total_score": float(
                best_metrics["total_score"]
            ),
            "components": {
                "gpu_demand_distance": float(
                    best_metrics[
                        "gpu_demand_distance"
                    ]
                ),
                "runtime_cdf_distance": float(
                    best_metrics[
                        "runtime_cdf_distance"
                    ]
                ),
                "interarrival_distance": float(
                    best_metrics[
                        "interarrival_distance"
                    ]
                ),
                "burst_fraction_distance": float(
                    best_metrics[
                        "burst_fraction_distance"
                    ]
                ),
                "gpu_service_distance": float(
                    best_metrics[
                        "gpu_service_distance"
                    ]
                ),
            },
            "weights": normalized_weights,
            "target_profile": target_profile,
            "selected_profile": candidate_profile,
        },
    }

    return window, metadata