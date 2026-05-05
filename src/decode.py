import numpy as np
import random

# f: Per is the permutation that maps i to f[i]
Per = list[int]

char_prob: np.ndarray = np.genfromtxt("data/letter_probabilities.csv", delimiter=',')
trans_mat: np.ndarray = np.genfromtxt("data/letter_transition_matrix.csv", delimiter=',')
alph_size: int = len(char_prob)
# Precomputed log versions for fast LL evaluation (errstate suppresses log(0) warning)
with np.errstate(divide='ignore'):
    _log_char_prob: np.ndarray = np.where(char_prob > 0, np.log(char_prob), -np.inf)
    _log_trans_mat: np.ndarray = np.where(trans_mat > 0, np.log(trans_mat), -np.inf)
alphabet: dict[str, int] = {}
alphabet_rev: list[str] = []
with open("data/alphabet.csv") as f:
    idx = 0
    for char in f.readline()[:-1].split(','):
        alphabet_rev.append(char)
        alphabet[char] = idx
        idx += 1

def inv(f: Per) -> Per:
    """ Computes inverse of a permutation. """
    ret: Per = [0 for i in range(alph_size)]
    for idx in range(alph_size):
        ret[f[idx]] = idx
    return ret

def log_likelihood(y: str, f: Per) -> float:
    """ Computes the log likelihood of y given f. """ 
    if len(y) == 0:
        return 0.0
    inv_f: Per = inv(f)
    prob = char_prob[inv_f[alphabet[y[0]]]]
    if prob == 0:
        return -np.inf
    log_start_prob = np.log(prob)
    log_trans_probs = (
        np.log(trans_mat[inv_f[alphabet[y[k]]]][inv_f[alphabet[y[k-1]]]]) if 
        trans_mat[inv_f[alphabet[y[k]]]][inv_f[alphabet[y[k-1]]]] != 0 else
        -np.inf
        for k in range(1, len(y))
    )
    return log_start_prob + sum(log_trans_probs)

def _identity_per() -> Per:
    return [i for i in range(alph_size)]

def _freq_init(ciphertext: str) -> Per:
    """Initialize encoder by matching ciphertext character frequencies to English."""
    counts = np.zeros(alph_size)
    for c in ciphertext:
        counts[alphabet[c]] += 1
    cipher_by_freq: list[int] = list(np.argsort(-counts))
    english_by_freq: list[int] = list(np.argsort(-char_prob))
    per: Per = [0] * alph_size
    for eng_idx, cip_idx in zip(english_by_freq, cipher_by_freq):
        per[eng_idx] = cip_idx
    return per

def _ll_fast(B: np.ndarray, ci_0: int, inv_per: np.ndarray) -> float:
    """O(a²) log-likelihood using precomputed bigram counts and numpy indexing.

    Equivalent to log_likelihood(ciphertext, per) but ~100x faster because it
    avoids the O(n) Python loop: the transition sum becomes a single matrix
    multiply using the precomputed bigram count matrix B.
    """
    first = float(_log_char_prob[inv_per[ci_0]])
    if not np.isfinite(first):
        return float('-inf')
    # trans_mat convention is trans_mat[current][previous], so for bigram (prev, cur):
    # contribution = log_trans_mat[inv_per[cur]][inv_per[prev]]
    # np.ix_(inv_per, inv_per)[a,b] = log_trans_mat[inv_per[a]][inv_per[b]]
    # → we need [cur, prev] ordering, which is the transpose of [prev, cur] = B ordering.
    perm_log_T = _log_trans_mat[np.ix_(inv_per, inv_per)].T  # now [prev, cur] aligned with B
    mask = B > 0
    if mask.any() and not np.all(np.isfinite(perm_log_T[mask])):
        return float('-inf')
    return first + float(np.sum(B[mask] * perm_log_T[mask]))

def _greedy_climb(B: np.ndarray, ci_0: int, inv_per: np.ndarray) -> tuple[np.ndarray, float]:
    """Greedy hill-climbing on inv_per using O(a²) LL per swap evaluation."""
    inv_per = inv_per.copy()
    cur_score = _ll_fast(B, ci_0, inv_per)
    for _ in range(alph_size):
        improved = False
        for c_p in range(alph_size):
            for c_q in range(c_p + 1, alph_size):
                inv_per[c_p], inv_per[c_q] = inv_per[c_q], inv_per[c_p]
                new_score = _ll_fast(B, ci_0, inv_per)
                if new_score > cur_score:
                    cur_score = new_score
                    improved = True
                else:
                    inv_per[c_p], inv_per[c_q] = inv_per[c_q], inv_per[c_p]
        if not improved:
            break
    return inv_per, cur_score

_FULL_BURN_IN = 10000
_FULL_ITERATIONS = 50000
_GREEDY_RESTARTS = 50

def map_estimate(ciphertext: str, burn_in: int = _FULL_BURN_IN, num_iterations: int = _FULL_ITERATIONS) -> Per:
    """ Given we observed a stirng ciphertext, return the MAP estimator f that
        may have generated it. Uses the Metropolis-Hastings MCMC algorithm.

        Internal LL uses precomputed bigram counts + numpy for O(a²) per evaluation
        instead of O(n), giving ~100x speedup that funds more restarts and iterations.

        Phase 1: frequency-based init → greedy hill-climb to a local optimum.
        Phase 2: multi-start greedy — perturb + re-climb 20 times, keeping the best.
        Phase 3: MH-MCMC from the best greedy warm start. """
    if len(ciphertext) == 0:
        return _identity_per()

    # Precompute bigram counts once — used by all subsequent LL evaluations
    ci = [alphabet[c] for c in ciphertext]
    B = np.zeros((alph_size, alph_size))
    for k in range(len(ci) - 1):
        B[ci[k], ci[k + 1]] += 1
    ci_0 = ci[0]

    # Phase 1: frequency-matched init → first greedy local optimum
    # Work in decoder (inv_per) space throughout for consistency with _ll_fast
    best_inv, best_score = _greedy_climb(B, ci_0, np.array(inv(_freq_init(ciphertext))))

    # Phase 2: multi-start greedy — perturb best decoder and re-climb
    for _ in range(_GREEDY_RESTARTS):
        perturbed = best_inv.copy()
        for _ in range(random.randint(2, 4)):
            c_p, c_q = random.sample(range(alph_size), 2)
            perturbed[c_p], perturbed[c_q] = perturbed[c_q], perturbed[c_p]
        restart_inv, score = _greedy_climb(B, ci_0, perturbed)
        if score > best_score:
            best_inv, best_score = restart_inv, score

    # Phase 3: MH-MCMC from the best greedy warm start
    cur_inv = best_inv.copy()
    cur_score = best_score
    for iter_no in range(num_iterations):
        if iter_no >= burn_in and cur_score > best_score:
            best_inv = cur_inv.copy()
            best_score = cur_score
        c_p, c_q = random.sample(range(alph_size), 2)
        cur_inv[c_p], cur_inv[c_q] = cur_inv[c_q], cur_inv[c_p]
        new_score = _ll_fast(B, ci_0, cur_inv)
        accept_log_prob = min(0.0, new_score - cur_score) if np.isfinite(cur_score) else 0.0
        if np.log(random.random()) < accept_log_prob:
            cur_score = new_score
        else:
            cur_inv[c_p], cur_inv[c_q] = cur_inv[c_q], cur_inv[c_p]

    return inv(list(best_inv))

def _plaintext_under_encoder(segment: str, encoder: Per) -> str:
    if len(segment) == 0:
        return ""
    decoder: Per = inv(encoder)
    return "".join(alphabet_rev[decoder[alphabet[c]]] for c in segment)

def _bigram_mi_scores(ciphertext: str) -> np.ndarray:
    """Compute MI_left(s) + MI_right(s) for every split position s in O(n * alph_size^2).

    Mutual information between adjacent characters is permutation-invariant:
    a single-cipher segment scores MI ≈ English MI regardless of which permutation
    was used. A segment that mixes two different ciphers scores lower because the
    transition structure is inconsistent. So argmax over s gives the breakpoint.

    Prefix bigram counts let us evaluate all n+1 splits without repeating work.
    """
    n = len(ciphertext)
    a = alph_size
    ci = [alphabet[c] for c in ciphertext]

    # cum_bg[s] = bigram count matrix for ciphertext[:s]
    # bigrams are pairs (ci[k], ci[k+1]) for k = 0..s-2 (entirely within the segment)
    cum_bg = np.zeros((n + 1, a, a))
    for s in range(2, n + 1):
        cum_bg[s] = cum_bg[s - 1]
        cum_bg[s, ci[s - 2], ci[s - 1]] += 1
    bg_total = cum_bg[n]

    def _mi(bg: np.ndarray, n_bg: int) -> float:
        if n_bg < 2:
            return 0.0
        p = bg / n_bg
        rows = bg.sum(axis=1, keepdims=True) / n_bg
        cols = bg.sum(axis=0, keepdims=True) / n_bg
        denom = rows * cols
        mask = (p > 0) & (denom > 0)
        return float(np.sum(p[mask] * np.log(p[mask] / denom[mask]))) if mask.any() else 0.0

    # Weighted score: n_bigrams * MI — equivalent to the LLR changepoint statistic.
    # Require MIN_SEG chars on each side so the empirical MI is estimated reliably.
    MIN_SEG = max(50, n // 10)
    scores = np.full(n + 1, -np.inf)
    for s in range(MIN_SEG, n - MIN_SEG + 1):
        n_l = s - 1
        n_r = n - s - 1
        bg_l = cum_bg[s]
        bg_r = bg_total - cum_bg[s + 1]
        scores[s] = n_l * _mi(bg_l, n_l) + n_r * _mi(bg_r, n_r)
    return scores

def _best_breakpoint_split(ciphertext: str) -> tuple[int, Per, Per]:
    """Locate split via MI scan, then refine using fixed-cipher LL evaluation.

    1. Weighted bigram MI scan across all valid positions — O(n * a²), no MCMC.
    2. Run full MCMC once at the MI candidate to estimate enc_L, enc_R.
    3. Refine split by evaluating those FIXED ciphers at every position in a wide
       window — O(radius * n), no extra MCMC. This works because for positions
       close to the true breakpoint the ciphers are approximately correct, and the
       LL correctly identifies where the split should be.
    """
    n = len(ciphertext)

    cand = int(np.argmax(_bigram_mi_scores(ciphertext)))

    enc_l = map_estimate(ciphertext[:cand], burn_in=_FULL_BURN_IN, num_iterations=_FULL_ITERATIONS)
    enc_r = map_estimate(ciphertext[cand:], burn_in=_FULL_BURN_IN, num_iterations=_FULL_ITERATIONS)

    # Fixed-cipher sweep: no MCMC, just LL evaluation at each candidate split.
    radius = max(100, int(np.sqrt(n)) * 3)
    lo = max(0, cand - radius)
    hi = min(n, cand + radius)
    best_s = cand
    best_ll = log_likelihood(ciphertext[:cand], enc_l) + log_likelihood(ciphertext[cand:], enc_r)
    for s in range(lo, hi + 1):
        if s == cand:
            continue
        ll = log_likelihood(ciphertext[:s], enc_l) + log_likelihood(ciphertext[s:], enc_r)
        if ll > best_ll:
            best_ll = ll
            best_s = s

    return best_s, enc_l, enc_r

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    if not has_breakpoint:
        encoder: Per = map_estimate(ciphertext)
        return _plaintext_under_encoder(ciphertext, encoder)

    split, enc_left, enc_right = _best_breakpoint_split(ciphertext)
    return (
        _plaintext_under_encoder(ciphertext[:split], enc_left)
        + _plaintext_under_encoder(ciphertext[split:], enc_right)
    )
