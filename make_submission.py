"""
This script zips up the contents of the directory it is in into a format
that is ready to submit to Gradescope.

To make sure this script works properly, it should be run from the same
directory that you run test.py from.

Usage: python3 make-zip.py

WARNING: We recommend you don't modify this file.
"""

import os
import shutil
import tempfile

if os.path.exists("submission.zip"):
    print("Removing old submission.zip...")
    os.remove("submission.zip")

print("Creating submission.zip ready to upload to Gradescope...")
with tempfile.TemporaryDirectory() as tmp_dir_path:
    # Use temporary directory to avoid this issue:
    #   https://stackoverflow.com/q/49467850/1337463

    # Code taken from https://stackoverflow.com/a/25650295/1337463
    shutil.make_archive(base_name=f"{tmp_dir_path}/submission",
                        format="zip",
                        root_dir=".")

    shutil.move(src=f"{tmp_dir_path}/submission.zip", dst="submission.zip")
