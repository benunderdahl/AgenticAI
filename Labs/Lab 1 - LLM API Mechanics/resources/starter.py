"""
04/07 Lab Starter — SEC Filing Analysis System

This file provides the project structure and imports for the Day 01 lab.
Implement each exercise in its own file (or section) as described in
the lab handout. This starter gives you the shared configuration and
data loading utilities.

Run generate_sample_filings.py first to create the sample data.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---

DATA_DIR = Path(__file__).parent / "sec-filings"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. Create a .env file with your key."
    )


# --- Data Loading Utilities ---

def load_filing(filename: str) -> str:
    """Load a single SEC filing text file."""
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Filing not found: {filepath}. Run generate_sample_filings.py first."
        )
    return filepath.read_text(encoding="utf-8")


def load_all_filings() -> list[dict]:
    """Load all filings listed in the manifest."""
    manifest_path = DATA_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run generate_sample_filings.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        entry["content"] = load_filing(entry["filename"])
    return manifest


def load_filings_by_type(filing_type: str) -> list[dict]:
    """Load filings of a specific type (e.g., '10-K', '8-K')."""
    all_filings = load_all_filings()
    return [f for f in all_filings if f["filing_type"] == filing_type]


# --- Quick verification ---

if __name__ == "__main__":
    try:
        filings = load_all_filings()
        print(f"Loaded {len(filings)} filings from {DATA_DIR}")
        for f in filings[:5]:
            print(f"  {f['filename']}: {f['filing_type']} | "
                  f"{f['company']} | {len(f['content']):,} chars")
        if len(filings) > 5:
            print(f"  ... and {len(filings) - 5} more")
    except FileNotFoundError as e:
        print(f"Setup needed: {e}")
