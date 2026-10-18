"""Module-2 tests for replay isolation and diagnostic artifact semantics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import QuadraticProblem, RunConfig
from coordinate_geometry.adam_diagnostics import (
    canonical_gradient_history,
    diagnostic_records,
    main,
)


class AdamDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "study_status": "module2_adam_diagnostic",
            "protocol_version": 1,
            "seed": 22,
            "eigenvalues": [0.1, 1.0],
            "initial_loss": 1.0,
            "angles_degrees": [0, 45],
            "max_steps": 4,
            "checkpoints": [1, 2, 4],
            "gd_learning_rate": 0.5,
            "adam_learning_rate": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }

    def test_canonical_history_matches_exact_diagonal_gd_recurrence(self):
        eigenvalues = np.array([0.1, 1.0])
        problem = QuadraticProblem(eigenvalues, np.eye(2))
        initial = np.array([2.0, -3.0])
        config = RunConfig("gd", 0.5, 4)
        history = canonical_gradient_history(problem, initial, config)
        expected = np.asarray(
            [eigenvalues * ((1 - config.learning_rate * eigenvalues) ** step) * initial
             for step in range(config.max_steps)]
        )
        assert_allclose(history, expected, atol=1e-15)

    def test_identity_control_is_zero_and_general_rotation_has_a_defect(self):
        rows, problem, summary = diagnostic_records(self.settings)
        identity = [row for row in rows if row["angle_degrees"] == 0]
        rotated = [row for row in rows if row["angle_degrees"] == 45]
        assert_allclose([row["replay_relative_defect"] for row in identity], 0.0, atol=1e-14)
        assert_allclose([row["adam_normalized_position_error"] for row in identity], 0.0, atol=1e-14)
        self.assertGreater(max(row["replay_relative_defect"] for row in rotated), 0.01)
        self.assertGreater(max(row["adam_normalized_position_error"] for row in rotated), 0.01)
        self.assertEqual(len(rows), 8)
        self.assertEqual(problem["gradient_history_source"],
                         "canonical GD, gradient recorded before each update")
        self.assertFalse(summary["protocol_frozen"])

    def test_entrypoint_writes_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            output = root / "output"
            config.write_text(json.dumps(self.settings), encoding="utf-8")
            self.assertEqual(main(["--config", str(config), "--output", str(output)]), 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"diagnostics.csv", "problem.json", "run.log", "run_config.json", "summary.json"},
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["study_status"], "module2_adam_diagnostic")
            self.assertEqual(len(summary["angles"]), 2)

    def test_invalid_protocol_and_checkpoints_fail_loudly(self):
        invalid = dict(self.settings, study_status="main_study")
        with self.assertRaisesRegex(ValueError, "study_status"):
            diagnostic_records(invalid)
        invalid = dict(self.settings, checkpoints=[0, 4])
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            diagnostic_records(invalid)


    def test_discrete_settings_reject_coercion_and_nonfinite_values(self):
        invalid_values = (
            True, False, np.bool_(True), 1.9, 4.0, np.float64(2.0),
            float("nan"), float("inf"), float("-inf"), "4", None,
        )
        for field in ("max_steps", "seed", "checkpoints"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    replacement = [value] if field == "checkpoints" else value
                    invalid = dict(self.settings, **{field: replacement})
                    with self.assertRaisesRegex(ValueError, field):
                        diagnostic_records(invalid)

    def test_discrete_settings_enforce_ranges_and_unique_checkpoints(self):
        invalid_settings = (
            ("max_steps", 0), ("max_steps", -1), ("max_steps", np.int64(-4)),
            ("seed", -1), ("seed", np.int64(-1)),
            ("checkpoints", []), ("checkpoints", [1, 1]),
            ("checkpoints", [0, 4]), ("checkpoints", [-1, 4]),
            ("checkpoints", [1, 5]), ("checkpoints", None),
            ("checkpoints", 4), ("checkpoints", "14"),
            ("checkpoints", {1: True}), ("checkpoints", np.array(1)),
            ("checkpoints", np.array([[1, 4]])),
        )
        for field, value in invalid_settings:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    diagnostic_records(dict(self.settings, **{field: value}))

    def test_numpy_integer_settings_preserve_the_same_experiment(self):
        expected = diagnostic_records(self.settings)
        actual = diagnostic_records(dict(
            self.settings,
            max_steps=np.int64(4), seed=np.uint64(22),
            checkpoints=np.array([1, 2, 4], dtype=np.int32),
        ))
        self.assertEqual(actual, expected)
        self.assertIs(type(actual[2]["gd_config"]["max_steps"]), int)
        self.assertTrue(all(type(value) is int for value in actual[2]["checkpoints"]))

    def test_zero_and_large_integer_seeds_remain_supported(self):
        for seed in (0, np.uint64(0), 2**128 + 123):
            with self.subTest(seed=seed):
                rows, problem, _ = diagnostic_records(dict(self.settings, seed=seed))
                self.assertEqual(len(rows), 8)
                self.assertTrue(np.all(np.isfinite(problem["canonical_initial"])))

    def test_entrypoint_rejects_fractional_protocol_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, output = root / "config.json", root / "output"
            for field, value in (("max_steps", 4.9), ("checkpoints", [1.9, 4.2]), ("seed", 22.9)):
                with self.subTest(field=field):
                    config.write_text(json.dumps(dict(self.settings, **{field: value})), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        main(["--config", str(config), "--output", str(output)])
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
