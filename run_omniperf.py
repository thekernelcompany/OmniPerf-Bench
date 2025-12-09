#!/usr/bin/env python3
"""
OmniPerf-Bench - Performance optimization benchmark framework.

Usage:
    python run_omniperf.py config.yaml
    python run_omniperf.py config.yaml --dry-run
    python run_omniperf.py config.yaml -v

Examples:
    # Run with default config
    python run_omniperf.py configs/omniperf.yaml

    # Validate config without running
    python run_omniperf.py configs/omniperf.yaml --dry-run

    # Verbose output
    python run_omniperf.py configs/omniperf.yaml -v
"""
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omniperf.cli import main

if __name__ == "__main__":
    main()
