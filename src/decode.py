import numpy as np
import random

# f: Per is the permutation that maps i to f[i]
Per = list[int]

char_prob: np.ndarray = np.genfromtxt("data/letter_probabilities.csv", delimiter=',')
trans_mat: np.ndarray = np.genfromtxt("data/letter_transition_matrix.csv", delimiter=',')
alph_size: int = len(char_prob)
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

_FULL_BURN_IN = 10000
_FULL_ITERATIONS = 20000

def map_estimate(ciphertext: str, burn_in: int = _FULL_BURN_IN, num_iterations: int = _FULL_ITERATIONS) -> Per:
    """ Given we observed a stirng ciphertext, return the MAP estimator f that
        may have generated it. Uses the Metropolis-Hastings MCMC algorithm. """
    if len(ciphertext) == 0:
        return _identity_per()
    per: Per = _identity_per()
    best_per: Per = per.copy()
    best_score = log_likelihood(ciphertext, per)
    cur_score = best_score
    # empirically determine the number of iterations you need
    for iter_no in range(num_iterations):
        if iter_no >= burn_in:
            if cur_score > best_score:
                best_per = per.copy()
                best_score = cur_score
        new_per: Per = per.copy()
        i, j = random.sample(range(alph_size), 2) # choose two to randomly swap
        new_per[i], new_per[j] = new_per[j], new_per[i]
        # compute acceptance probability
        accept_log_prob = 0
        new_log_likelihood = log_likelihood(ciphertext, new_per)
        if cur_score != -np.inf:
            accept_log_prob = min(0, new_log_likelihood - cur_score) 
        else:
            accept_log_prob = 0
        accept = np.log(random.random()) < accept_log_prob
        if accept: # swap
            per = new_per
            cur_score = new_log_likelihood
    return best_per

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
