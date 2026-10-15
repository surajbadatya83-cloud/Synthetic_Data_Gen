import unittest

import pandas as pd

from streamlit_app.services.dataset_health import (
    DatasetInputError,
    analyze_dataset,
    parse_csv_bytes,
)


class ParseCsvTests(unittest.TestCase):
    def test_parses_csv_and_limits_rows(self):
        frame = parse_csv_bytes(b"x,y\n1,a\n2,b\n", max_rows=2)
        self.assertEqual(frame.shape, (2, 2))
        with self.assertRaisesRegex(DatasetInputError, "2-row"):
            parse_csv_bytes(b"x\n1\n2\n3\n", max_rows=2)

    def test_rejects_empty_and_header_only_uploads(self):
        with self.assertRaisesRegex(DatasetInputError, "empty"):
            parse_csv_bytes(b" \r\n")
        with self.assertRaisesRegex(DatasetInputError, "no data rows"):
            parse_csv_bytes(b"x,y\n")

    def test_rejects_malformed_csv(self):
        with self.assertRaises(DatasetInputError):
            parse_csv_bytes(b"x,y\n1,2\n3,4,5\n")
        with self.assertRaisesRegex(DatasetInputError, "duplicate column names"):
            parse_csv_bytes(b"x,x\n1,2\n")

    def test_rejects_too_many_columns(self):
        header = ",".join(f"c{i}" for i in range(501))
        with self.assertRaisesRegex(DatasetInputError, "500-column"):
            parse_csv_bytes((header + "\n" + ",".join(["1"] * 501)).encode())

    def test_rejects_row_by_column_cell_budget(self):
        with self.assertRaisesRegex(DatasetInputError, "cell analysis limit"):
            parse_csv_bytes(b"a,b,c\n1,2,3\n4,5,6\n", max_cells=3)

    def test_rejects_files_over_byte_limit(self):
        with self.assertRaisesRegex(DatasetInputError, "upload limit"):
            parse_csv_bytes(b"x\n1\n", max_bytes=2)


class DatasetHealthTests(unittest.TestCase):
    def test_clean_dataset_has_no_findings(self):
        frame = pd.DataFrame({
            "measurement": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["a", "b", "a", "b", "c"],
        })
        result = analyze_dataset(frame)
        self.assertEqual(result["summary"]["rows"], 5)
        self.assertEqual(result["summary"]["missing_cells"], 0)
        self.assertEqual(result["summary"]["duplicate_rows"], 0)
        self.assertTrue(result["findings"].empty)

    def test_missing_values_and_exact_duplicates_are_reported(self):
        frame = pd.DataFrame({"value": [1.0, 2.0, 2.0, None], "label": ["a", "b", "b", "c"]})
        result = analyze_dataset(frame)
        summary = result["summary"]
        self.assertEqual(summary["missing_cells"], 1)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["duplicate_percent"], 25.0)
        self.assertIn("Missing values", result["findings"]["category"].tolist())
        self.assertIn("Exact duplicates", result["findings"]["category"].tolist())

    def test_mixed_type_candidate_is_reported(self):
        frame = pd.DataFrame({"amount": ["12", "unknown", "14"]})
        result = analyze_dataset(frame)
        self.assertIn("Mixed values", result["findings"]["category"].tolist())
        self.assertEqual(result["columns"].loc[0, "inferred_type"], "mixed / inconsistent candidate")

    def test_integer_like_category_codes_skip_numerical_outlier_methods(self):
        result = analyze_dataset(pd.DataFrame({"code": ["1"] * 10 + ["2"]}))
        row = result["columns"].loc[0]
        self.assertEqual(row["inferred_type"], "low-cardinality numeric text (possibly categorical)")
        self.assertTrue(pd.isna(row["outlier_count"]))

    def test_constant_and_near_constant_columns_are_distinguished(self):
        frame = pd.DataFrame({"fixed": [7] * 50, "rarely_changes": [1] * 49 + [2]})
        result = analyze_dataset(frame)
        categories = result["findings"].groupby("column")["category"].apply(set).to_dict()
        self.assertIn("Constant column", categories["fixed"])
        self.assertIn("Near-constant column", categories["rarely_changes"])
        self.assertNotIn("Possible outliers (IQR heuristic)", categories["rarely_changes"])

    def test_iqr_outlier_is_flagged_without_calling_it_an_error(self):
        result = analyze_dataset(pd.DataFrame({"measure": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]}))
        row = result["columns"].set_index("column").loc["measure"]
        self.assertEqual(row["outlier_count"], 1)
        finding = result["findings"].query("category == 'Possible outliers (IQR heuristic)'").iloc[0]
        self.assertIn("not necessarily errors", finding["recommendation"])

    def test_high_cardinality_and_identifier_are_heuristics(self):
        result = analyze_dataset(pd.DataFrame({"record_code": [f"r{i}" for i in range(60)]}))
        findings = result["findings"]
        self.assertIn("High cardinality", findings["category"].tolist())
        possible_id = findings.query("category == 'Possible identifier'").iloc[0]
        self.assertIn("not a confirmed identifier", possible_id["finding"])

    def test_non_finite_values_are_reported_and_excluded_from_stats(self):
        result = analyze_dataset(pd.DataFrame({"measure": [1.0, 2.0, float("inf"), 4.0]}))
        self.assertIn("Non-finite numeric values", result["findings"]["category"].tolist())
        self.assertEqual(result["columns"].loc[0, "maximum"], 4.0)

    def test_small_text_only_dataset_and_all_missing_column_are_safe(self):
        frame = pd.DataFrame({"text": ["only one record"], "empty": [None]})
        result = analyze_dataset(frame)
        self.assertEqual(result["summary"]["rows"], 1)
        self.assertNotIn("numerical", result["columns"]["inferred_type"].tolist())
        self.assertIn("Empty column", result["findings"]["category"].tolist())

    def test_selected_target_distribution_and_imbalance(self):
        frame = pd.DataFrame({"target": [0] * 19 + [1], "feature": range(20)})
        result = analyze_dataset(frame, target_column="target")
        distribution = result["target_distribution"].set_index("class")
        self.assertEqual(distribution.loc["1", "count"], 1)
        self.assertIn("Target imbalance", result["findings"]["category"].tolist())
        without_target = analyze_dataset(frame)
        self.assertNotIn("Target imbalance", without_target["findings"]["category"].tolist())

    def test_analysis_does_not_mutate_input(self):
        frame = pd.DataFrame({"value": [1.0, 2.0, None, None], "group": ["a", "b", "c", "c"]})
        original = frame.copy(deep=True)
        result = analyze_dataset(frame)
        pd.testing.assert_frame_equal(frame, original)
        findings_csv = result["findings"].to_csv(index=False)
        self.assertIn("Missing values", findings_csv)
        self.assertIn("value", findings_csv)
        self.assertIn("Exact duplicates", findings_csv)

    def test_rejects_empty_dataframes(self):
        with self.assertRaisesRegex(DatasetInputError, "at least one row"):
            analyze_dataset(pd.DataFrame(columns=["x"]))


if __name__ == "__main__":
    unittest.main()
