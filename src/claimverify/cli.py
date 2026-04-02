"""CLI entry point for claimverify."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Scientific Claim Verification with RAG"
    )
    parser.add_argument(
        "command",
        choices=["eval-bm25", "eval-dense", "eval-hybrid", "version"],
        help="Command to run",
    )
    args = parser.parse_args()

    if args.command == "version":
        from claimverify import __version__
        print(f"claimverify v{__version__}")
    else:
        print(f"Run: python scripts/01_baseline_retrieval.py (for eval-bm25)")
        print(f"     python scripts/02_dense_retrieval.py (for eval-dense/hybrid)")


if __name__ == "__main__":
    main()
