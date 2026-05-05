import numpy as np
import random
import os as _os
import string as _string

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

_FULL_BURN_IN = 5000
_FULL_ITERATIONS = 30000
_GREEDY_RESTARTS = 30

# All disjoint pairs of swaps — precomputed once for the 2-swap neighbourhood search
_SWAP_PAIRS: list[tuple[int, int]] = [
    (p, q) for p in range(alph_size) for q in range(p + 1, alph_size)
]
_DISJOINT_TWO_SWAPS: list[tuple[int, int, int, int]] = [
    (p1, q1, p2, q2)
    for i, (p1, q1) in enumerate(_SWAP_PAIRS)
    for (p2, q2) in _SWAP_PAIRS[i + 1:]
    if len({p1, q1, p2, q2}) == 4   # swaps share no character
]

def _two_swap_refine(B: np.ndarray, ci_0: int, inv_per: np.ndarray) -> tuple[np.ndarray, float]:
    """Exhaustively search the disjoint 2-swap neighbourhood of inv_per.

    Single-swap greedy gets stuck when the global optimum requires two characters
    to be reassigned simultaneously (a local optimum under 1-swaps that is not
    a local optimum under 2-swaps). For a=28 there are ~61k disjoint pairs;
    at O(a²) per evaluation this takes ~300ms per outer loop iteration.
    After each improving 2-swap we re-run greedy to reach the new local optimum
    before searching again.
    """
    inv_per = inv_per.copy()
    cur_score = _ll_fast(B, ci_0, inv_per)
    while True:
        best_score = cur_score
        best_swap = None
        for p1, q1, p2, q2 in _DISJOINT_TWO_SWAPS:
            inv_per[p1], inv_per[q1] = inv_per[q1], inv_per[p1]
            inv_per[p2], inv_per[q2] = inv_per[q2], inv_per[p2]
            s = _ll_fast(B, ci_0, inv_per)
            if s > best_score:
                best_score = s
                best_swap = (p1, q1, p2, q2)
            inv_per[p2], inv_per[q2] = inv_per[q2], inv_per[p2]
            inv_per[p1], inv_per[q1] = inv_per[q1], inv_per[p1]
        if best_swap is None:
            break
        p1, q1, p2, q2 = best_swap
        inv_per[p1], inv_per[q1] = inv_per[q1], inv_per[p1]
        inv_per[p2], inv_per[q2] = inv_per[q2], inv_per[p2]
        inv_per, cur_score = _greedy_climb(B, ci_0, inv_per)
    return inv_per, cur_score

_TOP_LEVEL_RUNS = 2   # independent full runs; best LL across all is returned

def _map_estimate_once(ciphertext: str, B: np.ndarray, ci_0: int,
                       burn_in: int, num_iterations: int) -> tuple[np.ndarray, float]:
    """One complete run of phases 1-4. Returns (best_inv_per, best_score)."""
    # Phase 1: frequency init → greedy
    best_inv, best_score = _greedy_climb(B, ci_0, np.array(inv(_freq_init(ciphertext))))

    # Phase 2: multi-start greedy restarts
    for _ in range(_GREEDY_RESTARTS):
        perturbed = best_inv.copy()
        for _ in range(random.randint(2, 4)):
            c_p, c_q = random.sample(range(alph_size), 2)
            perturbed[c_p], perturbed[c_q] = perturbed[c_q], perturbed[c_p]
        restart_inv, score = _greedy_climb(B, ci_0, perturbed)
        if score > best_score:
            best_inv, best_score = restart_inv, score

    # Phase 3: MH-MCMC warm start
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

    # Phase 4: exhaustive disjoint 2-swap neighbourhood
    best_inv, best_score = _two_swap_refine(B, ci_0, best_inv)

    return best_inv, best_score

def map_estimate(ciphertext: str, burn_in: int = _FULL_BURN_IN, num_iterations: int = _FULL_ITERATIONS) -> Per:
    """ Given we observed a stirng ciphertext, return the MAP estimator f that
        may have generated it. Uses the Metropolis-Hastings MCMC algorithm.

        Runs _TOP_LEVEL_RUNS independent chains and returns the one with the
        highest log-likelihood, giving multiple chances to escape hard local optima. """
    if len(ciphertext) == 0:
        return _identity_per()

    ci = [alphabet[c] for c in ciphertext]
    B = np.zeros((alph_size, alph_size))
    for k in range(len(ci) - 1):
        B[ci[k], ci[k + 1]] += 1
    ci_0 = ci[0]

    best_inv, best_score = _map_estimate_once(ciphertext, B, ci_0, burn_in, num_iterations)
    for _ in range(_TOP_LEVEL_RUNS - 1):
        inv_candidate, score = _map_estimate_once(ciphertext, B, ci_0, burn_in, num_iterations)
        if score > best_score:
            best_inv, best_score = inv_candidate, score

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

    1. Weighted bigram MI scan — O(n * a²), no MCMC.
    2. Full MCMC at the MI candidate to estimate enc_L, enc_R.
    3. Fixed-cipher sweep over a window using _ll_fast with incrementally-updated
       prefix bigram matrices — O(radius * a²), independent of n.
       Previously used log_likelihood (O(n) Python loop × radius steps = O(n²) total);
       this replaces it with O(a²) numpy per step = O(radius * a²) total.
    """
    n = len(ciphertext)
    ci = [alphabet[c] for c in ciphertext]
    a = alph_size

    cand = int(np.argmax(_bigram_mi_scores(ciphertext)))

    enc_l = map_estimate(ciphertext[:cand], burn_in=_FULL_BURN_IN, num_iterations=_FULL_ITERATIONS)
    enc_r = map_estimate(ciphertext[cand:], burn_in=_FULL_BURN_IN, num_iterations=_FULL_ITERATIONS)
    inv_l = np.array(inv(enc_l))
    inv_r = np.array(inv(enc_r))

    radius = max(100, int(np.sqrt(n)) * 3)
    lo = max(0, cand - radius)
    hi = min(n, cand + radius)

    # Build bg_total once: all bigrams in ciphertext
    bg_total = np.zeros((a, a))
    for k in range(n - 1):
        bg_total[ci[k], ci[k + 1]] += 1

    # cum_bg_s represents the bigram count matrix for ciphertext[:s]
    # (bigrams k = 0..s-2 that lie entirely within the left segment).
    # Initialise to cum_bg[lo]: bigrams k = 0..lo-2.
    cum_bg_s = np.zeros((a, a))
    for k in range(lo - 1):
        cum_bg_s[ci[k], ci[k + 1]] += 1

    # bg_right = bigrams entirely within ciphertext[s:] = bg_total - cum_bg[s+1]
    # cum_bg[s+1] = cum_bg[s] + bigram k=s-1.  Initialise for s=lo.
    bg_right = (bg_total - cum_bg_s).copy()
    if 0 < lo < n:
        bg_right[ci[lo - 1], ci[lo]] -= 1   # remove straddle bigram k=lo-1

    best_s = lo
    best_ll = -np.inf
    for s in range(lo, hi + 1):
        ll_l = _ll_fast(cum_bg_s, ci[0], inv_l) if s > 0 else 0.0
        ll_r = _ll_fast(bg_right, ci[s],  inv_r) if s < n else 0.0
        ll = ll_l + ll_r
        if ll > best_ll:
            best_ll = ll
            best_s = s

        # Advance to s+1:
        #   cum_bg[s+1] = cum_bg[s] + bigram k=s-1
        #   bg_right_{s+1} = bg_right_s − bigram k=s
        if 0 < s < n:
            cum_bg_s[ci[s - 1], ci[s]] += 1
        if s < n - 1:
            bg_right[ci[s], ci[s + 1]] -= 1

    return best_s, enc_l, enc_r

# ---------------------------------------------------------------------------
# Dictionary-based post-processing
# ---------------------------------------------------------------------------

def _load_word_set() -> frozenset[str]:
    # Path is relative to this file's parent directory (project root)
    word_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'words_alpha.txt')
    try:
        with open(word_file) as f:
            words = frozenset(line.strip().lower() for line in f if line.strip().isalpha())
        return words | {'a', 'i'}   # ensure common single-letter words are present
    except FileNotFoundError:
        return frozenset()

_WORD_SET: frozenset[str] = _load_word_set()

def _dictionary_refine(decoded: str) -> str:
    """Greedily swap character pairs to maximise the number of dictionary words.

    After MCMC decoding, rare characters (e.g. 'j' vs 'q') are sometimes swapped
    because they occur too infrequently for the bigram statistics to distinguish
    them. The substitution cipher maps every instance of a character identically,
    so one global swap of the two confused characters fixes all words at once.

    For each non-dictionary word we try all C(26,2)=325 simultaneous pairwise
    swaps (using str.translate, which swaps both directions atomically) and vote
    for whichever swap fixes the most words. We apply the winning swap if at least
    2 words agree, then repeat until no confident swap remains.
    """
    if not _WORD_SET:
        return decoded

    from collections import Counter
    letters = _string.ascii_lowercase
    for _ in range(10):          # at most 10 rounds; converges in 1–2 in practice
        words = [tok.rstrip('.') for tok in decoded.split(' ') if tok.rstrip('.')]
        char_counts = Counter(decoded)

        # Safety constraint: only consider swapping characters that are RARE
        # (appear ≤ 5 times in the decoded text). Common characters like 'e' or 't'
        # are never confused by a correct MCMC run; swapping them would corrupt the
        # output. Rare characters (e.g. 'q' appearing 3 times) are plausible confusions.
        rare = {c for c in letters if char_counts.get(c, 0) <= 5}
        if not rare:
            break

        # Net score = words fixed − words broken.
        # Many swaps can "fix" a non-dict word (e.g. "qust"→"dust","gust","just"…),
        # but only the correct swap avoids breaking currently-valid words.
        net: dict[tuple[str, str], int] = {}
        for i, c in enumerate(letters):
            for c2 in letters[i + 1:]:
                # At least one character in the pair must be rare
                if c not in rare and c2 not in rare:
                    continue
                tbl = str.maketrans(c + c2, c2 + c)
                affected = [w for w in words if c in w or c2 in w]
                if not affected:
                    continue
                fixed  = sum(1 for w in affected if w not in _WORD_SET
                             and (t := w.translate(tbl)) != w and t in _WORD_SET)
                broken = sum(1 for w in affected if w     in _WORD_SET
                             and w.translate(tbl) not in _WORD_SET)
                score = fixed - broken
                if score > 0:
                    net[(c, c2)] = score

        if not net:
            break
        best = max(net, key=lambda k: net[k])
        if net[best] < 2:        # require net gain of at least 2 words
            break
        c1, c2 = best
        decoded = decoded.translate(str.maketrans(c1 + c2, c2 + c1))

    return decoded

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    if not has_breakpoint:
        encoder: Per = map_estimate(ciphertext)
        result = _plaintext_under_encoder(ciphertext, encoder)
    else:
        split, enc_left, enc_right = _best_breakpoint_split(ciphertext)
        result = (
            _plaintext_under_encoder(ciphertext[:split], enc_left)
            + _plaintext_under_encoder(ciphertext[split:], enc_right)
        )
    return _dictionary_refine(result)
