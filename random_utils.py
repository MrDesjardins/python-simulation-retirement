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
    m = len(returns)
    available = np.arange(m)
    sequence = []
    neg_streak = pos_streak = 0
    cumulative_drop = cumulative_rise = 0.0

    while len(sequence) < n_years and len(available) > 0:
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

        # If no valid candidate found, relax constraints instead of ignoring
        if not accepted:
            neg_streak = pos_streak = 0
            cumulative_drop = cumulative_rise = 0.0

    return np.array(sequence[:n_years], dtype=np.int64)
