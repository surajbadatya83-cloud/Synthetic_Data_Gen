"""Dataframe-only health checks; these helpers never write input or report files."""

from __future__ import annotations

import csv
from io import BytesIO, TextIOWrapper
from typing import Any

import numpy as np
import pandas as pd


MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_DATA_ROWS = 250_000
MAX_COLUMNS = 500
MAX_CELLS = 5_000_000
FINDING_COLUMNS = ["category", "severity", "column", "finding", "recommendation"]


class DatasetInputError(ValueError):
    """Raised when an uploaded CSV cannot be safely analyzed."""


def parse_csv_bytes(
    content: bytes,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_rows: int = MAX_DATA_ROWS,
    max_cells: int = MAX_CELLS,
) -> pd.DataFrame:
    """Parse a bounded UTF-8 CSV upload and reject empty or oversized inputs."""
    if not content or not content.strip():
        raise DatasetInputError("The uploaded file is empty. Choose a CSV containing a header and data rows.")
    if len(content) > max_bytes:
        raise DatasetInputError(f"The file exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")

    try:
        header_stream = TextIOWrapper(BytesIO(content), encoding="utf-8-sig", newline="")
        try:
            header = next(csv.reader(header_stream, strict=True))
        finally:
            header_stream.close()
        if not header:
            raise DatasetInputError("The CSV does not contain a header row.")
        if len(header) > MAX_COLUMNS:
            raise DatasetInputError(f"The CSV exceeds the {MAX_COLUMNS}-column limit.")
        allowed_rows = min(max_rows, max_cells // len(header))
        if allowed_rows < 1:
            raise DatasetInputError(f"The CSV exceeds the {max_cells:,}-cell analysis limit.")
        duplicate_headers = pd.Index(header).duplicated()
        if duplicate_headers.any():
            duplicates = pd.Index(header)[duplicate_headers].astype(str).tolist()
            raise DatasetInputError(f"The CSV contains duplicate column names: {', '.join(duplicates)}")
        frame = pd.read_csv(BytesIO(content), encoding="utf-8-sig", on_bad_lines="error", low_memory=False, nrows=allowed_rows + 1)
    except DatasetInputError:
        raise
    except csv.Error as error:
        raise DatasetInputError(f"Could not parse the CSV header: {error}") from error
    except pd.errors.EmptyDataError as error:
        raise DatasetInputError("The CSV has no readable header or data.") from error
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise DatasetInputError(f"Could not parse this CSV: {error}") from error

    if frame.shape[1] == 0:
        raise DatasetInputError("The CSV does not contain any columns.")
    if frame.empty:
        raise DatasetInputError("The CSV contains a header but no data rows.")
    if len(frame) > allowed_rows:
        if allowed_rows == max_rows:
            raise DatasetInputError(f"The CSV exceeds the {max_rows:,}-row analysis limit.")
        raise DatasetInputError(f"The CSV exceeds the {max_cells:,}-cell analysis limit.")
    if frame.columns.has_duplicates:
        duplicate_headers = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise DatasetInputError(f"The CSV contains duplicate column names: {', '.join(duplicate_headers)}")
    return frame


def _add_finding(
    findings: list[dict[str, str]],
    category: str,
    severity: str,
    column: str,
    finding: str,
    recommendation: str,
) -> None:
    findings.append({
        "category": category,
        "severity": severity,
        "column": column,
        "finding": finding,
        "recommendation": recommendation,
    })


def _date_parse_ratio(series: pd.Series) -> float:
    values = series.dropna().astype(str)
    if values.empty:
        return 0.0
    # Avoid interpreting plain numeric text (including year-like numbers) as dates.
    if pd.to_numeric(values, errors="coerce").notna().all():
        return 0.0
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return float(parsed.notna().mean())


def analyze_dataset(frame: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    """Compute reproducible facts and explicitly heuristic warnings for a dataframe."""
    if not isinstance(frame, pd.DataFrame):
        raise DatasetInputError("The selected input is not a tabular dataset.")
    if frame.empty or frame.shape[1] == 0:
        raise DatasetInputError("The dataset must contain at least one row and one column.")
    if frame.columns.has_duplicates:
        raise DatasetInputError("Column names must be unique to produce an unambiguous report.")
    if target_column is not None and target_column not in frame.columns:
        raise DatasetInputError("The selected target column is not present in the dataset.")

    row_count, column_count = frame.shape
    missing_by_column = frame.isna().sum()
    duplicate_rows = int(frame.duplicated().sum())
    findings: list[dict[str, str]] = []
    column_results: list[dict[str, Any]] = []

    if duplicate_rows:
        duplicate_percent = duplicate_rows / row_count * 100
        _add_finding(
            findings, "Exact duplicates", "warning", "(dataset)",
            f"{duplicate_rows:,} exact duplicate row(s) ({duplicate_percent:.2f}% of rows).",
            "Review the repeated rows and determine whether they are valid repeated observations before removing anything.",
        )

    for column in frame.columns:
        series = frame[column]
        non_null = series.dropna()
        non_null_count = len(non_null)
        unique_count = int(non_null.nunique(dropna=True))
        missing_count = int(missing_by_column[column])
        missing_percent = missing_count / row_count * 100
        dtype_name = str(series.dtype)
        inferred_type = "other"
        numeric_candidate = False
        continuous_candidate = True
        outlier_count: int | None = None
        outlier_percent: float | None = None
        skew_value: float | None = None

        if non_null_count == 0:
            inferred_type = "empty"
            _add_finding(
                findings, "Empty column", "warning", str(column),
                "The column contains no non-missing values.",
                "Check the source export or remove the column only after confirming it is not needed.",
            )
        elif pd.api.types.is_bool_dtype(series.dtype):
            inferred_type = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            inferred_type = "datetime"
        elif pd.api.types.is_numeric_dtype(series.dtype):
            numeric_candidate = True
            if pd.api.types.is_integer_dtype(series.dtype) and unique_count <= 10:
                inferred_type = "low-cardinality numeric (possibly categorical)"
                continuous_candidate = False
                _add_finding(
                    findings, "Data type review", "info", str(column),
                    f"The numeric column has only {unique_count} distinct value(s); it may contain category codes.",
                    "Confirm whether these values represent measurements or encoded categories before applying numerical methods.",
                )
            else:
                inferred_type = "numerical"
        else:
            numeric_values = pd.to_numeric(non_null, errors="coerce")
            numeric_ratio = float(numeric_values.notna().mean()) if non_null_count else 0.0
            if 0 < numeric_ratio < 1:
                inferred_type = "mixed / inconsistent candidate"
                _add_finding(
                    findings, "Mixed values", "warning", str(column),
                    f"{numeric_values.notna().sum():,} of {non_null_count:,} non-missing values parse as numbers; the rest do not.",
                    "Inspect the non-numeric entries and standardize the column only after confirming their intended meaning.",
                )
            elif numeric_ratio == 1:
                numeric_candidate = True
                numeric_values = numeric_values.astype(float)
                numeric_integral = bool(
                    np.isfinite(numeric_values.to_numpy()).all()
                    and np.equal(numeric_values, np.floor(numeric_values)).all()
                )
                if unique_count <= 10 and numeric_integral:
                    inferred_type = "low-cardinality numeric text (possibly categorical)"
                    continuous_candidate = False
                    _add_finding(
                        findings, "Data type review", "info", str(column),
                        f"All values parse as numbers, but there are only {unique_count} integer-like values and the column was read as text.",
                        "Confirm whether these are category codes or measurements before converting or applying numerical methods.",
                    )
                else:
                    inferred_type = "numeric stored as text"
                    _add_finding(
                        findings, "Data type review", "info", str(column),
                        "All non-missing values parse as numbers, but the CSV inferred a text type.",
                        "Check for formatting such as spaces or separators, then consider an explicit numeric conversion in a separate cleaned copy.",
                    )
            elif _date_parse_ratio(non_null) >= 0.9:
                inferred_type = "datetime candidate stored as text"
                _add_finding(
                    findings, "Data type review", "info", str(column),
                    "Most non-missing text values parse as dates or times.",
                    "Confirm the intended date format and timezone before converting this column.",
                )
            else:
                inferred_type = "categorical / text"

        if missing_count:
            _add_finding(
                findings, "Missing values", "warning", str(column),
                f"{missing_count:,} missing value(s) ({missing_percent:.2f}% of rows).",
                "Investigate why values are missing; choose an imputation or exclusion policy only after considering the data source and analysis goal.",
            )

        if non_null_count and unique_count == 1:
            _add_finding(
                findings, "Constant column", "info", str(column),
                "All non-missing values are identical.",
                "Confirm the field is expected to vary; constant features usually add no information to a model.",
            )
        elif non_null_count >= 20 and unique_count > 1:
            top_fraction = float(non_null.value_counts(normalize=True, dropna=True).iloc[0])
            if top_fraction >= 0.98:
                _add_finding(
                    findings, "Near-constant column", "info", str(column),
                    f"One value accounts for {top_fraction * 100:.2f}% of non-missing values (heuristic threshold: 98%).",
                    "Check whether this feature is meaningful despite its low variation.",
                )

        if inferred_type != "empty" and non_null_count >= 50:
            cardinality = unique_count / non_null_count
            if unique_count >= 20 and cardinality >= 0.80:
                _add_finding(
                    findings, "High cardinality", "info", str(column),
                    f"{unique_count:,} distinct values among {non_null_count:,} non-missing values ({cardinality * 100:.1f}%).",
                    "Review whether this is a free-text field or a possible identifier before treating it as a category.",
                )
            if unique_count >= 20 and cardinality >= 0.95:
                _add_finding(
                    findings, "Possible identifier", "info", str(column),
                    f"{cardinality * 100:.1f}% of non-missing values are distinct; this is an identifier-like heuristic, not a confirmed identifier.",
                    "Check whether values identify records or people; exclude identifiers from modeling when they do not represent a meaningful feature.",
                )

        if numeric_candidate:
            parsed_numeric = pd.to_numeric(non_null, errors="coerce").dropna().astype(float)
            non_finite_count = int((~np.isfinite(parsed_numeric)).sum())
            if non_finite_count:
                _add_finding(
                    findings, "Non-finite numeric values", "warning", str(column),
                    f"{non_finite_count:,} non-finite numeric value(s), such as positive or negative infinity, were detected.",
                    "Inspect the source calculations or export; non-finite values are excluded from numerical summaries.",
                )
            values = parsed_numeric[np.isfinite(parsed_numeric)]
            if len(values):
                q1, median, q3 = (float(values.quantile(q)) for q in (0.25, 0.5, 0.75))
                iqr = q3 - q1
                minimum, maximum = float(values.min()), float(values.max())
                if continuous_candidate and len(values) >= 4 and iqr > 0:
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_count = int(((values < lower) | (values > upper)).sum())
                    outlier_percent = outlier_count / len(values) * 100
                    if outlier_count:
                        _add_finding(
                            findings, "Possible outliers (IQR heuristic)", "info", str(column),
                            f"{outlier_count:,} value(s) fall outside [{lower:g}, {upper:g}] using 1.5 × IQR.",
                            "Inspect the distribution and source records; statistical outliers are not necessarily errors.",
                        )
                if continuous_candidate and len(values) >= 8:
                    skew_value = float(values.skew())
                    if np.isfinite(skew_value) and abs(skew_value) >= 1:
                        _add_finding(
                            findings, "Skewed distribution", "info", str(column),
                            f"Sample skewness is {skew_value:.2f} (heuristic flag at |skewness| ≥ 1).",
                            "Inspect the histogram and consider whether the skew is expected before transforming values.",
                        )
                if (values < 0).any():
                    negative_count = int((values < 0).sum())
                    _add_finding(
                        findings, "Negative values to validate", "info", str(column),
                        f"{negative_count:,} negative numeric value(s) are present; this is a fact, not a domain error by itself.",
                        "Confirm whether negative values are valid for this field using its documented meaning or schema.",
                    )
                numeric_summary = {
                    "minimum": minimum,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "maximum": maximum,
                    "mean": float(values.mean()),
                    "std": float(values.std()) if len(values) > 1 else None,
                    "skewness": skew_value,
                }
            else:
                numeric_summary = {}
        else:
            numeric_summary = {}

        column_results.append({
            "column": str(column),
            "dtype": dtype_name,
            "inferred_type": inferred_type,
            "non_missing": non_null_count,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_percent": unique_count / non_null_count * 100 if non_null_count else 0.0,
            "outlier_count": outlier_count,
            "outlier_percent": outlier_percent,
            **numeric_summary,
        })

    target_distribution = pd.DataFrame(columns=["class", "count", "percent"])
    if target_column:
        target = frame[target_column].dropna()
        if target.empty:
            _add_finding(
                findings, "Target imbalance", "warning", str(target_column),
                "The selected target has no non-missing labels.",
                "Check the target column source before using it for classification.",
            )
        else:
            counts = target.value_counts(dropna=False)
            target_distribution = pd.DataFrame({
                "class": counts.index.astype(str),
                "count": counts.to_numpy(dtype=int),
                "percent": (counts.to_numpy(dtype=float) / len(target) * 100).round(2),
            })
            if len(counts) == 1:
                _add_finding(
                    findings, "Target imbalance", "warning", str(target_column),
                    "The selected target contains only one observed class.",
                    "Verify the selected target and source data; a one-class target cannot support ordinary classification evaluation.",
                )
            elif float(counts.min() / len(target)) < 0.10:
                _add_finding(
                    findings, "Target imbalance", "info", str(target_column),
                    f"The smallest class represents {float(counts.min() / len(target) * 100):.2f}% of non-missing labels (heuristic flag below 10%).",
                    "Review class proportions and choose evaluation or resampling methods appropriate to the task.",
                )

    findings_frame = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    columns_frame = pd.DataFrame(column_results)
    total_cells = row_count * column_count
    missing_cells = int(missing_by_column.sum())
    affected_columns = findings_frame.loc[
        findings_frame["column"] != "(dataset)", "column"
    ].nunique() if not findings_frame.empty else 0
    outlier_values = sum(
        int(value) for value in columns_frame["outlier_count"] if pd.notna(value)
    ) if not columns_frame.empty else 0

    return {
        "summary": {
            "rows": row_count,
            "columns": column_count,
            "missing_cells": missing_cells,
            "missing_percent": missing_cells / total_cells * 100 if total_cells else 0.0,
            "duplicate_rows": duplicate_rows,
            "duplicate_percent": duplicate_rows / row_count * 100 if row_count else 0.0,
            "columns_with_findings": int(affected_columns),
            "outlier_values": outlier_values,
            "findings_count": len(findings_frame),
        },
        "columns": columns_frame,
        "findings": findings_frame,
        "target_distribution": target_distribution,
    }
