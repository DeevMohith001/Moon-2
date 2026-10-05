"""Script to create output directories and archive stale results"""
import pathlib
import shutil
from datetime import datetime

root = pathlib.Path(".")
outputs = root / "outputs"
results = root / "results"

# Create required output directories
for d in [
    outputs / "reference_paper",
    outputs / "teacher",
    outputs / "distillation",
    outputs / "student" / "regression",
    outputs / "student" / "classification",
    outputs / "benchmarks",
    outputs / "ablations",
    outputs / "error_analysis",
    outputs / "figures" / "precision_recall",
    outputs / "finance",
    outputs / "archive",
    results / "metrics",
    results / "figures",
    results / "predictions",
    results / "error_analysis",
]:
    d.mkdir(parents=True, exist_ok=True)
    print(f"Created: {d}")

# Archive stale synthetic results
archive_dir = outputs / "archive"
archived = []

for src in results.glob("*.csv"):
    dest = archive_dir / ("LEGACY_" + src.name)
    shutil.copy2(src, dest)
    archived.append(src.name)
    print(f"Archived to outputs/archive/: {src.name}")

# Create archive README
lines = [
    "# Legacy / Stale Results Archive",
    "",
    "All files in this directory are **LEGACY ARTIFACTS** generated from synthetic development data.",
    "",
    "## Status: NOT FOR FINAL RESULTS",
    "",
    "These files were generated from `generate_development_dataset()` (synthetic data).",
    "They must NOT be consumed by the final dashboard or cited as research results.",
    "",
    "## Archived Files",
    "",
    "| File | Original Location | Reason |",
    "|---|---|---|",
]
for f in archived:
    lines.append(f"| `LEGACY_{f}` | `results/{f}` | Synthetic data - stale |")
lines += [
    "",
    f"## Archived On: {datetime.now().isoformat()}",
    "",
    "## How to Generate Real Results",
    "",
    "1. Load real data: `python -m src.data.ingestion`",
    "2. Run preprocessing: `python -m src.data.preprocessing`",
    "3. Run teacher labeling: `python -m src.paper_pipeline --provider openai`",
    "4. Run benchmarks: see README.md",
    "",
]
archive_readme = archive_dir / "ARCHIVE_README.md"
archive_readme.write_text("\n".join(lines), encoding="utf-8")

print(f"\nArchived {len(archived)} files to outputs/archive/")
print("Created archive README")

# Create reference_paper directory with paper numbers
ref_dir = outputs / "reference_paper"
import json
paper_numbers = {
    "note": "These are the numbers FROM THE ORIGINAL PAPER (Deng et al. 2023).",
    "source": "arXiv:2212.11311 Table 1 and Table 2",
    "warning": "These are NOT our reproduction results. Our results are in results/ directory.",
    "teacher_llm_6shot_cot_k8": {
        "model": "PaLM-540B",
        "FiQA-News": {"accuracy": 0.759, "macro_f1": 0.752},
        "FiQA-Post": {"accuracy": 0.714, "macro_f1": 0.695},
    },
    "finbert_prosus": {
        "model": "ProsusAI/finbert",
        "FiQA-News": {"accuracy": 0.690, "macro_f1": 0.652},
        "FiQA-Post": {"accuracy": 0.641, "macro_f1": 0.608},
    },
    "student_regression": {
        "model": "Charformer-102M (MSE distillation)",
        "FiQA-News": {"accuracy": 0.776, "macro_f1": 0.770},
        "FiQA-Post": {"accuracy": 0.735, "macro_f1": 0.718},
    },
}

(ref_dir / "paper_reported_numbers.json").write_text(
    json.dumps(paper_numbers, indent=2), encoding="utf-8"
)
print("Created reference_paper/paper_reported_numbers.json")
