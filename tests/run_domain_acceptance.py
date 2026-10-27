"""Run deterministic domain cases and write an evidence-labelled report; no remote calls."""
import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_domain_acceptance import DomainAcceptanceTests, records

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DomainAcceptanceTests))
report = {"test_kind": "deterministic_model_standin_with_real_tools", "live_model_tested": False,
    "a1_5_tested": False, "domain_cases": records, "accepted_cases": sum(row["domain_accepted"] for row in records),
    "total_cases": len(records), "acceptance_rate": sum(row["domain_accepted"] for row in records) / len(records) if records else None,
    "limitations": ["Scripted decisions do not prove live model reasoning, tool selection or unattended long-task reliability", "No provider usage, price or retry telemetry; corresponding metrics are unknown"]}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
raise SystemExit(0 if result.wasSuccessful() else 1)
