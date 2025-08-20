import json, pathlib, re, subprocess, csv

# -- Global Paths -- #
VLLM_DIR = pathlib.Path("~/coding-mess/vllm").expanduser()
JSON_FILE = pathlib.Path("data/vllm-gso-problems/divided/vllm_results_part_3.json")
OUT_CSV = pathlib.Path("vllm_classification_review.csv")

# -- Helper Functions -- #
def rg(pattern: str, path: pathlib.Path, glob: str) -> list[str]:
    """Returns the list filenames that match the pattern under global path"""
    cmd = [
        "rg", "-i", "-l",
        "--glob", glob, 
        "--max-filesize", "1M",
        "--fixed-strings", pattern,
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        return out.splitlines()
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            return []
        raise

TEST_GLOBS = ["tests/**"]
BENCH_GLOBS = ["benchmarks/**", "benchmarks*"]

def seen_in_repo(clues: set[str], globs: list[str]) -> bool:
    """True if any clue string appears i at least one file restricted by globs"""
    for clue in clues:
        if len(clue) < 3:
            continue
        for g in globs:
            if rg(clue, VLLM_DIR, g):
                return True
    return False

# ---------------------------------------------------------------------------
# Load the JSON file
# ---------------------------------------------------------------------------
print(f"Loading classification results from {JSON_FILE} …")
with JSON_FILE.open() as f:
    records = json.load(f)["classification_results"]
print(f"Loaded {len(records)} commits")

# ---------------------------------------------------------------------------
# Evaluate each commit
# ---------------------------------------------------------------------------
rows = []
for rec in records:
    # Build a set of clue words from file stems and affected API tokens
    clues: set[str] = set()

    # 1) filename stems
    for path_str in rec["files_changed"]:
        stem = pathlib.Path(path_str).stem.lower()
        if stem:
            clues.add(stem)

    # 2) API fragments
    for api in rec.get("affected_apis", []):
        for token in re.split(r"[._]", api):
            token_lc = token.lower()
            if token_lc:
                clues.add(token_lc)

    # Limit token length to avoid accidental matches
    clues = {c for c in clues if 3 <= len(c) <= 40}

    # Search repo once per commit
    tests_exist   = seen_in_repo(clues, TEST_GLOBS)
    benches_exist = seen_in_repo(clues, BENCH_GLOBS)
    sample_clues  = ", ".join(sorted(clues)[:3])  # at most three clues for the CSV

    rows.append({
        "commit_hash"                 : rec["commit_hash"],
        "category"                    : rec["category"],
        "json_has_tests"              : rec["has_tests"],
        "json_has_benchmarks"         : rec["has_benchmarks"],
        "repo_related_tests_exist"    : tests_exist,
        "repo_related_benchmarks_exist": benches_exist,
        "sample_clues"                : sample_clues,
    })

# ---------------------------------------------------------------------------
# Write CSV report
# ---------------------------------------------------------------------------
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✓ Audit report written to {OUT_CSV.resolve()}") 