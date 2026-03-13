from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence


@dataclass
class SplitReport:
    """
    Result container for one split.
    """

    split_name: str
    train_domains: list[str]
    test_domain: str
    metrics: dict[str, float]
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "train_domains": self.train_domains,
            "test_domain": self.test_domain,
            "metrics": self.metrics,
            "notes": self.notes,
        }


@dataclass
class BenchmarkReport:
    """
    Container for a full benchmark evaluation.
    """

    benchmark_name: str
    reports: list[SplitReport] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_report(self, report: SplitReport) -> None:
        self.reports.append(report)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "metadata": self.metadata,
            "reports": [r.to_dict() for r in self.reports],
            "aggregate": aggregate_reports(self.reports),
        }


def aggregate_reports(
    reports: Sequence[SplitReport],
) -> dict[str, Any]:
    """
    Aggregate metrics across splits.
    """
    if len(reports) == 0:
        return {
            "num_splits": 0,
            "metrics_mean": {},
            "metrics_std": {},
        }

    metric_names = sorted(
        {
            metric_name
            for report in reports
            for metric_name in report.metrics.keys()
        }
    )

    metrics_mean: dict[str, float] = {}
    metrics_std: dict[str, float] = {}

    for metric_name in metric_names:
        values = [
            float(report.metrics[metric_name])
            for report in reports
            if metric_name in report.metrics
        ]

        mean_value = sum(values) / len(values)
        std_value = (sum((v - mean_value) ** 2 for v in values) / len(values)) ** 0.5

        metrics_mean[metric_name] = mean_value
        metrics_std[metric_name] = std_value

    return {
        "num_splits": len(reports),
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std,
    }


class ReportWriter:
    """
    Writer for benchmark reports.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        report: BenchmarkReport,
        filename: str = "benchmark_report.json",
    ) -> Path:
        path = self.root / filename

        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        return path

    def write_jsonl(
        self,
        report: BenchmarkReport,
        filename: str = "split_reports.jsonl",
    ) -> Path:
        path = self.root / filename

        with path.open("w", encoding="utf-8") as f:
            for split_report in report.reports:
                f.write(json.dumps(split_report.to_dict(), ensure_ascii=False) + "\n")

        return path

    def write_summary_txt(
        self,
        report: BenchmarkReport,
        filename: str = "summary.txt",
    ) -> Path:
        path = self.root / filename
        aggregate = aggregate_reports(report.reports)

        lines = [
            f"Benchmark: {report.benchmark_name}",
            f"Number of splits: {aggregate['num_splits']}",
            "",
            "Mean metrics:",
        ]

        for metric_name, value in aggregate["metrics_mean"].items():
            std = aggregate["metrics_std"].get(metric_name, 0.0)
            lines.append(f"  {metric_name}: {value:.6f} ± {std:.6f}")

        lines.append("")
        lines.append("Per-split results:")

        for split_report in report.reports:
            lines.append(
                f"  - {split_report.split_name}: "
                f"train={split_report.train_domains}, "
                f"test={split_report.test_domain}, "
                f"metrics={split_report.metrics}"
            )

        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return path