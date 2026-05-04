import os
import numpy as np
import random
import matplotlib.pyplot as plt

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

GRAPHS_DIR = "graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

def save_and_show(filename: str):
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()

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
        np.log(trans_mat[inv_f[alphabet[y[k]]]][inv_f[alphabet[y[k-1]]]])
        if trans_mat[inv_f[alphabet[y[k]]]][inv_f[alphabet[y[k-1]]]] != 0 else -np.inf
        for k in range(1, len(y))
    )
    return log_start_prob + sum(log_trans_probs)

def decode_with_encoder(ciphertext: str, encoder: Per) -> str:
    decoder: Per = inv(encoder)
    plaintext = "".join(alphabet_rev[decoder[alphabet[char]]] for char in ciphertext)
    return plaintext

def accuracy(decoded: str, plaintext: str) -> float:
    return sum(decoded[i] == plaintext[i] for i in range(len(plaintext))) / len(plaintext)

def sliding_acceptance_rate(accepted: list[int], T: int) -> list[float]:
    rates = []
    for t in range(len(accepted)):
        left = max(0, t - T + 1)
        window = accepted[left:t+1]
        rates.append(sum(window) / len(window))
    return rates

def map_estimate(ciphertext: str, true_plaintext: str | None = None):
    """ Given we observed a string ciphertext, return the MAP estimator f that
        may have generated it. Uses the Metropolis-Hastings MCMC algorithm.

        Also records traces for plotting.
    """
    per: Per = [i for i in range(alph_size)]
    best_per: Per = per.copy()
    best_score = log_likelihood(ciphertext, per)
    cur_score = best_score

    BURN_IN = 10_000
    NUM_ITERATIONS = 20_000

    # traces
    log_likelihood_trace = []
    accepted_trace = []
    accuracy_trace = []
    bits_per_symbol_trace = []

    for iter_no in range(NUM_ITERATIONS):
        if iter_no >= BURN_IN:
            if cur_score > best_score:
                best_per = per.copy()
                best_score = cur_score

        new_per: Per = per.copy()
        i, j = random.sample(range(alph_size), 2)  # choose two to randomly swap
        new_per[i], new_per[j] = new_per[j], new_per[i]

        new_log_likelihood = log_likelihood(ciphertext, new_per)
        accept_log_prob = min(0.0, new_log_likelihood - cur_score)

        accept = np.log(random.random()) < accept_log_prob
        if accept:
            per = new_per
            cur_score = new_log_likelihood
            accepted_trace.append(1)
        else:
            accepted_trace.append(0)

        log_likelihood_trace.append(cur_score)
        bits_per_symbol_trace.append(-cur_score / (len(ciphertext) * np.log(2)))

        if true_plaintext is not None:
            decoded = decode_with_encoder(ciphertext, per)
            accuracy_trace.append(accuracy(decoded, true_plaintext))

    return {
        "best_per": best_per,
        "best_score": best_score,
        "log_likelihood_trace": log_likelihood_trace,
        "accepted_trace": accepted_trace,
        "accuracy_trace": accuracy_trace,
        "bits_per_symbol_trace": bits_per_symbol_trace,
        "burn_in": BURN_IN,
        "num_iterations": NUM_ITERATIONS,
    }

def decode(ciphertext: str, has_breakpoint: bool, true_plaintext: str | None = None) -> str:
    result = map_estimate(ciphertext, true_plaintext=true_plaintext)
    encoder: Per = result["best_per"]
    decoder: Per = inv(encoder)
    plaintext = "".join(alphabet_rev[decoder[alphabet[char]]] for char in ciphertext)
    return plaintext

# -------------------------
# Plotting functions
# -------------------------

def plot_log_likelihood(result):
    x = np.arange(1, len(result["log_likelihood_trace"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, result["log_likelihood_trace"])
    plt.axvline(result["burn_in"], linestyle='--')
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood of accepted state vs iteration")
    save_and_show("log_likelihood.png")

def plot_acceptance_rate(result, T: int = 1000):
    rates = sliding_acceptance_rate(result["accepted_trace"], T)
    x = np.arange(1, len(rates) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, rates)
    plt.axvline(result["burn_in"], linestyle='--')
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel(f"Acceptance rate (window T={T})")
    plt.title("Sliding-window acceptance rate vs iteration")
    save_and_show("acceptance_rate.png")

def plot_accuracy(result):
    if len(result["accuracy_trace"]) == 0:
        raise ValueError("Need true_plaintext to plot decoding accuracy.")

    x = np.arange(1, len(result["accuracy_trace"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, result["accuracy_trace"])
    plt.axvline(result["burn_in"], linestyle='--')
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Decoding accuracy")
    plt.title("Decoding accuracy vs iteration")
    save_and_show("accuracy.png")

def plot_bits_per_symbol(result, english_entropy_bits: float | None = None):
    x = np.arange(1, len(result["bits_per_symbol_trace"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(x, result["bits_per_symbol_trace"], label="Negative log-likelihood per symbol")
    plt.axvline(result["burn_in"], linestyle='--')
    if english_entropy_bits is not None:
        plt.axhline(english_entropy_bits, linestyle='--', label=f"English entropy ≈ {english_entropy_bits}")
        plt.legend()
    plt.xscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Bits per symbol")
    plt.title("Negative log-likelihood per symbol vs iteration")
    save_and_show("bits_per_symbol.png")

# -------------------------
# Segment experiment
# -------------------------

def split_into_segments(s: str, seg_len: int) -> list[str]:
    return [s[i:i+seg_len] for i in range(0, len(s), seg_len)]

def segment_experiment(ciphertext: str, plaintext: str, segment_lengths: list[int]):
    results = []

    for seg_len in segment_lengths:
        cipher_segments = split_into_segments(ciphertext, seg_len)
        plain_segments = split_into_segments(plaintext, seg_len)

        decoded_segments = []
        for cseg, pseg in zip(cipher_segments, plain_segments):
            result = map_estimate(cseg, true_plaintext=pseg)
            decoded_segments.append(decode_with_encoder(cseg, result["best_per"]))

        decoded_full = "".join(decoded_segments)
        acc = accuracy(decoded_full, plaintext)

        results.append((seg_len, acc))
        print(f"segment length = {seg_len}, accuracy = {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in results], [x[1] for x in results], marker='o')
    plt.xscale("log")
    plt.xlabel("Segment length")
    plt.ylabel("Decoding accuracy")
    plt.title("Accuracy vs segment length")
    save_and_show("segment_experiment.png")

    return results

if __name__ == "__main__":
    plaintext = ""
    ciphertext = ""

    with open("./data/sample/plaintext.txt") as f:
        plaintext = f.readline()

    with open("./data/sample/ciphertext.txt") as f:
        ciphertext = f.readline()

    result = map_estimate(ciphertext, true_plaintext=plaintext)

    plot_log_likelihood(result)             # (a)
    plot_acceptance_rate(result, T=1000)    # (b)
    plot_accuracy(result)                   # (c)
    plot_bits_per_symbol(result, english_entropy_bits=1.3)  # (e)

    segment_experiment(ciphertext, plaintext, [25, 50, 100, 200, 500, len(ciphertext)])  # (d)
