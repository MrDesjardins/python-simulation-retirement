import numpy as np


def generate_block_bootstrap_indices(
    rng: np.random.Generator,
    total_years: int,
    n_years: int,
    block_size: int = 5,
    circular: bool = True,
) -> np.ndarray:
    """
    Generate n_years indices using fixed-length block bootstrap.

    Why this matters for retirement Monte Carlo
    -------------------------------------------
    The naive approach — picking n_years individual years at random from
    history — destroys the autocorrelation that exists in real markets.
    Specifically, it cannot produce multi-year crash regimes that have
    historically driven the worst retirement outcomes:

        1929-1932 (Great Depression)   : -8.4%, -25.1%, -43.8%, -8.6%
        1973-1974 (stagflation)        : -14.7%, -26.5%
        2000-2002 (dot-com)            : -9.1%, -11.9%, -22.1%
        2007-2009 (GFC)                : -36.6%

    Block bootstrap samples consecutive blocks of historical years (here,
    blocks of `block_size` years), keeping these crash sequences intact.
    Each scenario draws ceil(n_years / block_size) blocks with replacement.

    Args:
        rng: numpy Generator
        total_years: length of the historical returns array (e.g. 154)
        n_years: how many years to generate (e.g. 40)
        block_size: length of each consecutive block (default 5 ≈ typical
            business-cycle persistence length used in financial econometrics)
        circular: if True, blocks wrap around the end of the historical
            series so every starting year has equal probability (Politis &
            Romano 1992 circular block bootstrap). Recommended.

    Returns:
        np.ndarray of shape (n_years,) with int64 indices into the returns
        array.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if total_years <= 0:
        raise ValueError(f"total_years must be positive, got {total_years}")
    if n_years <= 0:
        raise ValueError(f"n_years must be positive, got {n_years}")

    n_blocks = (n_years + block_size - 1) // block_size  # ceil

    if circular:
        starts = rng.integers(0, total_years, size=n_blocks)
        offsets = np.arange(block_size)
        idx_2d = (starts[:, None] + offsets[None, :]) % total_years
    else:
        max_start = total_years - block_size
        if max_start < 0:
            raise ValueError(
                f"block_size ({block_size}) cannot exceed total_years ({total_years}) "
                "when circular=False"
            )
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        offsets = np.arange(block_size)
        idx_2d = starts[:, None] + offsets[None, :]

    return idx_2d.ravel()[:n_years].astype(np.int64)


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
