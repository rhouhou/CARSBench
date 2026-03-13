from __future__ import annotations

import matplotlib.pyplot as plt

from CARSBench.benchmark.reports import BenchmarkReport


def plot_metric_bar(
    report: BenchmarkReport,
    metric_name: str = "rmse",
) -> None:
    """
    Bar plot of one metric across benchmark splits.
    """
    names = [split_report.split_name for split_report in report.reports]
    values = [split_report.metrics[metric_name] for split_report in report.reports]

    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel(metric_name)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_metric_by_test_domain(
    report: BenchmarkReport,
    metric_name: str = "rmse",
) -> None:
    """
    Plot metric values grouped by test domain.
    """
    domains = [split_report.test_domain for split_report in report.reports]
    values = [split_report.metrics[metric_name] for split_report in report.reports]

    plt.figure(figsize=(8, 4))
    plt.bar(domains, values)
    plt.ylabel(metric_name)
    plt.xlabel("Test domain")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()