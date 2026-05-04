"""
Script for testing your decoder on sample data.
If you pass the tests here, your code probably won't crash on Gradescope.
Usage: python3 test.py
WARNING: We recommend you don't modify this file.
"""

from typing import List, NamedTuple, Optional, Tuple

import logging
import os
import platform
import signal
import subprocess
import sys
import time
import traceback


def first_line(file_path):
    # Return first line of file as string, without trailing newline
    with open(file_path) as f:
        return f.readline().rstrip("\r\n")


class RunResult(NamedTuple):
    stdout: str = ""
    stderr: str = ""
    elapsed_secs: float = 0.0
    crash_reason: Optional[str] = None  # None means no crash


def run_decode_cli(
    executable_path: str,
    ciphertext: str,
    has_breakpoint: bool,
    command_prefix: Optional[List[str]] = None,
    timeout_secs: Optional[float] = None,
) -> RunResult:
    """
    return: output, elapsed_secs, crash_reason
    An output of None indicates a crash.
    """
    if command_prefix is None:
        command_prefix = []

    if not os.path.exists(executable_path):
        logging.error("decode-cli does not exist")
        return RunResult(crash_reason="decode-cli does not exist")

    # Explicitly call python on windows
    if platform.system() == "Windows":
        command_prefix.append(sys.executable)
    else:
        # Otherwise ensure executable can be executed
        subprocess.call(["chmod", "+x", executable_path])

    executable_dir = os.path.dirname(
        executable_path)  # foo/bar/decode-cli -> foo/bar
    executable_file = os.path.basename(
        executable_path)  # foo/bar/decode-cli -> decode-cli

    crash_reason = None
    start_time_secs = time.monotonic()

    # Based off of https://stackoverflow.com/a/36955420/1337463
    with subprocess.Popen(
            cwd=executable_dir,
            args=command_prefix + [
                f"./{executable_file}",
                ciphertext,
                str(has_breakpoint),
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="UTF-8",
    ) as process:
        try:
            process_outputs = process.communicate(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            # Terminate with extreme prejudice
            # (ref: https://stackoverflow.com/a/4047975/1337463)
            logging.error("decode-cli timed out")
            logging.error("killing TLEd process group...")
            os.killpg(process.pid, signal.SIGKILL)
            logging.error("killed TLEd process group")

            crash_reason = "time limit exceeded"
            process_outputs = process.communicate()

        stdout, stderr = (o.strip("\r\n") for o in process_outputs)

        if crash_reason is None and process.returncode != 0:
            logging.error("decode-cli failed")
            logging.error(f"return code: {process.returncode}")
            logging.error("STDERR:")
            logging.error(stderr)

            crash_reason = "generic crash"

    end_time_secs = time.monotonic()
    elapsed_secs = end_time_secs - start_time_secs

    if crash_reason is None and len(stdout) != len(ciphertext):
        logging.error("Decoded output must have same length as ciphertext")
        logging.error(f"Output     length: {len(stdout)}")
        logging.error(f"Ciphertext length: {len(ciphertext)}")
        logging.error(f"Output:")
        logging.error(f"\"{stdout}\"")
        crash_reason = "Decoded output has invalid length"

    return RunResult(
        stdout=stdout,
        stderr=stderr,
        elapsed_secs=elapsed_secs,
        crash_reason=crash_reason,
    )


def count_matches(a: str, b: str) -> int:
    """Returns the number of locations where the two strings are equal."""
    assert len(a) == len(b)
    return sum(int(i == j) for i, j in zip(a, b))


def fail_if_crash(rr: RunResult):
    if rr.crash_reason is None:
        return

    print()
    print()
    print("!!! ERROR !!!")
    print(f"Reason: {rr.crash_reason}")
    print("Your code seems to have errors.",
          "Please fix them and then rerun this test.")
    exit(-1)


def main():
    from src.encode import assert_clean

    logging.basicConfig(format="%(levelname)s - %(message)s")

    executable_path = "./decode-cli"
    plaintext = first_line("data/sample/short_plaintext.txt")
    ciphertext = first_line("data/sample/short_ciphertext.txt")
    ciphertext_with_breakpoint = first_line(
        "data/sample/short_ciphertext_breakpoint.txt")
    dummy_text = "the quick brown fox jumped over the lazy dog."

    assert_clean(plaintext)
    assert_clean(dummy_text)

    print("Running no breakpoint test...")
    res = run_decode_cli(executable_path, ciphertext, False)
    fail_if_crash(res)
    print(
        f"Score (no breakpoint): {count_matches(plaintext, res.stdout)} out of {len(plaintext)}"
    )
    print(f"Elapsed secs (no breakpoint): {res.elapsed_secs}")
    print()

    print("Running breakpoint test...")
    res = run_decode_cli(executable_path, ciphertext_with_breakpoint, True)
    fail_if_crash(res)
    print(
        f"Score (breakpoint): {count_matches(plaintext, res.stdout)} out of {len(plaintext)}"
    )
    print(f"Elapsed secs (breakpoint): {res.elapsed_secs}")
    print()

    print("Checking that you are not hardcoding inputs...")
    res = run_decode_cli(executable_path, dummy_text, False)
    fail_if_crash(res)
    count_matches(dummy_text, res.stdout)
    print()

    print("SUCCESS")
    print("decode-cli ran succesfully on sample data.")


if __name__ == "__main__":
    main()