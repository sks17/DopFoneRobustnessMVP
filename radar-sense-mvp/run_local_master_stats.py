"""Run a lightweight local benchmark sweep to get a general feel for the data before massive HPC trials.

Rough local estimate:
- The current MVP stack evaluates about 40 to 50 waveform trials per second on this machine.
- That suggests roughly 48,000 to 60,000 trials could fit into twenty minutes.

This script intentionally stays lightweight and config-driven. It is meant for quick local inspection of master
statistics, not for exhaustive robustness sweeps.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_clean_dataset import main as build_clean_dataset_main
from scripts.build_perturbed_dataset import main as build_perturbed_dataset_main
from scripts.run_benchmark import main as run_benchmark_main
from scripts.summarize_results import main as summarize_results_main, read_json_file


# Spec:
# - General description: Run the local clean-build, perturbed-build, benchmark, and summary pipeline in sequence.
# - Params: None.
# - Pre: Repository configs are present and valid.
# - Post: Writes the benchmark artifacts and prints the final summary.
# - Mathematical definition: Not applicable; this is a top-level orchestration script.
def main() -> None:
    """Run the full local benchmark pipeline and print the master summary."""
    build_clean_dataset_main()
    build_perturbed_dataset_main()
    run_benchmark_main()
    summarize_results_main()
    summary_path = PROJECT_ROOT / "data" / "manifests" / "benchmark_summary.json"
    summary = read_json_file(summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
