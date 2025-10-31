import numpy as np


def generate_constrained_indices(
    rng, returns, n_years, max_consec_neg=4, max_consec_drop=-0.5
):
    m = len(returns)
    available = np.arange(m)
    sequence = []
    neg_streak = 0
    cumulative_drop = 0.0

    while len(sequence) < n_years and len(available) > 0:
        candidate_idx = rng.integers(0, len(available))
        candidate = available[candidate_idx]
        r = returns[candidate]

        # Check constraints
        if r < 0:
            if neg_streak + 1 > max_consec_neg:
                continue
            if cumulative_drop + r <= max_consec_drop:
                continue
            neg_streak += 1
            cumulative_drop += r
        else:
            neg_streak = 0
            cumulative_drop = 0.0

        # Accept this year
        sequence.append(candidate)
        available = np.delete(available, candidate_idx)  # remove index from pool

    # Fallback: if we ran out early (due to too many constraints), pad randomly
    if len(sequence) < n_years:
        remaining = rng.choice(available, n_years - len(sequence), replace=False)
        sequence.extend(remaining)

    return np.array(sequence, dtype=np.int64)
