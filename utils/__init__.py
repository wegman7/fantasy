"""Utility package for fantasy ADP vs ECR analysis.

Modules:
- paths: Path builders for data files.
- loaders: CSV loading helpers.
- cleaners: DataFrame cleaning and normalization.
- merge: Merge datasets into unified frame.
- metrics: Error metrics and rankings.
- plotting: Visualization utilities.
"""

from . import paths, loaders, cleaners, merge, metrics, plotting  # noqa: F401

__all__ = ["paths", "loaders", "cleaners", "merge", "metrics", "plotting"]
