import numpy as np


def generate_constrained_indices(
    rng,
    returns,
    n_years,
    max_consec_neg=4,
    max_consec_drop=-0.85,   # -85%
    max_consec_pos=9,
    max_consec_rise=1.32,    # +132%
    max_attempts_per_step=50
):
    """
    Generate a sequence of n_years indices from returns array with real-life constraints.

    Constraints prevent unrealistic sequences:
    - Maximum consecutive negative years
    - Maximum cumulative decline during negative streaks
    - Maximum consecutive positive years
    - Maximum cumulative rise during positive streaks

    If constraints cannot be satisfied, they are progressively relaxed to ensure
    exactly n_years indices are always returned.

    Returns:
        np.ndarray: Array of exactly n_years indices (int64)
    """
    m = len(returns)

    # Input validation
    if n_years <= 0:
        raise ValueError(f"n_years must be positive, got {n_years}")
    if n_years > m:
        raise ValueError(f"n_years ({n_years}) cannot exceed available returns ({m})")

    available = np.arange(m)
    sequence = []
    neg_streak = pos_streak = 0
    cumulative_drop = cumulative_rise = 0.0

    while len(sequence) < n_years:
        # Safety check: if we've exhausted available indices, sample with replacement
        if len(available) == 0:
            # This should rarely happen, but prevents crashes
            # Sample from all returns (with replacement) for remaining slots
            remaining = n_years - len(sequence)
            fallback_indices = rng.choice(m, size=remaining, replace=True)
            sequence.extend(fallback_indices.tolist())
            break

        attempts = 0
        accepted = False

        while attempts < max_attempts_per_step and not accepted:
            attempts += 1
            candidate_idx = rng.integers(0, len(available))
            candidate = available[candidate_idx]
            r = returns[candidate]

            if r < 0:
                if neg_streak + 1 > max_consec_neg:
                    continue
                if cumulative_drop + r < max_consec_drop:  # STRICT limit
                    continue

                # Accept
                neg_streak += 1
                pos_streak = 0
                cumulative_drop += r
                cumulative_rise = 0.0

            else:
                if pos_streak + 1 > max_consec_pos:
                    continue
                if cumulative_rise + r > max_consec_rise:  # STRICT limit
                    continue

                # Accept
                pos_streak += 1
                neg_streak = 0
                cumulative_rise += r
                cumulative_drop = 0.0

            # Candidate accepted
            sequence.append(candidate)
            available = np.delete(available, candidate_idx)
            accepted = True

        # If no valid candidate found after max_attempts, relax constraints and pick any
        if not accepted:
            # Reset streaks and pick a random candidate from available pool
            neg_streak = pos_streak = 0
            cumulative_drop = cumulative_rise = 0.0

            # Pick any remaining candidate
            candidate_idx = rng.integers(0, len(available))
            candidate = available[candidate_idx]
            sequence.append(candidate)
            available = np.delete(available, candidate_idx)

    # Ensure we return exactly n_years elements
    result = np.array(sequence[:n_years], dtype=np.int64)
    assert len(result) == n_years, f"Expected {n_years} indices, got {len(result)}"
    return result
