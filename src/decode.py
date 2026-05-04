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

def map_estimate(ciphertext: str) -> Per:
    """ Given we observed a stirng ciphertext, return the MAP estimator f that
        may have generated it. Uses the Metropolis-Hastings MCMC algorithm. """
    per: Per = [i for i in range(alph_size)]
    best_per: Per = per.copy()
    best_score = log_likelihood(ciphertext, per)
    cur_score = best_score
    # empirically determine the number of iterations you need
    BURN_IN = 10000
    # First BURN_IN iterations are not counted
    NUM_ITERATIONS = 20000
    for iter_no in range(NUM_ITERATIONS):
        if iter_no >= BURN_IN:
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

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    encoder: Per = map_estimate(ciphertext)
    decoder: Per = inv(encoder)
    plaintext = "".join(alphabet_rev[decoder[alphabet[char]]] for char in ciphertext)
    return plaintext
