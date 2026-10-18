"""Independently rebuild Module 1 artifacts using second-order position recurrences.

Does not import the production runner, optimizer updates or spectral helpers.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


def verify(output):
    # Imported callers and the CLI share this entry point. Invalidate prior
    # success before reading any input so every failed audit leaves no marker.
    marker = output / "artifact_verification.json"
    marker.unlink(missing_ok=True)
    settings = json.loads((output / "run_config.json").read_text())
    saved = json.loads((output / "trajectories.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert hashlib.sha256((output / "run_config.json").read_bytes()).hexdigest() == summary["config_sha256"]
    with (output / "summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    d = np.asarray(settings["eigenvalues"], dtype=float)
    mu, largest = min(d), max(d)
    beta_opt = ((np.sqrt(largest) - np.sqrt(mu)) / (np.sqrt(largest) + np.sqrt(mu)))**2
    expected_candidates = [
        ("gd_inverse_L", "gd", 1 / largest, 0.0),
        ("gd_optimal", "gd", 2 / (mu + largest), 0.0),
        ("gd_boundary", "gd", 2 / largest, 0.0),
        ("gd_unstable", "gd", 2.1 / largest, 0.0),
        ("hb_optimal", "heavy_ball", 4 / (np.sqrt(largest) + np.sqrt(mu))**2, beta_opt),
        ("hb_boundary", "heavy_ball", 2 * (1 + beta_opt) / largest, beta_opt),
        ("hb_unstable", "heavy_ball", 2.1 * (1 + beta_opt) / largest, beta_opt),
    ]
    u = np.random.default_rng(settings["seed"]).normal(size=len(d))
    z0 = np.sqrt(2 / (u @ (d * u))) * u
    assert_allclose(saved["eigenvalues"], d, atol=0, rtol=0)
    assert_allclose(saved["canonical_initial"], z0, atol=0, rtol=0)
    schedule = [(angle, candidate) for angle in settings["angles_degrees"]
                for candidate in expected_candidates]
    assert len(saved["cases"]) == len(schedule) == len(csv_rows) == summary["cases"]
    atol, rtol = settings["verification_atol"], settings["verification_rtol"]
    max_reconstruction_error = 0.0
    for case, (angle, (name, algorithm, eta, beta)), csv_row in zip(saved["cases"], schedule, csv_rows):
        assert case["candidate"] == name and case["angle_degrees"] == angle
        assert case["config"]["algorithm"] == algorithm
        assert case["config"]["learning_rate"] == eta and case["config"]["beta"] == beta
        for key in ("max_steps", "tolerance", "explosive_threshold"):
            assert case["config"][key] == settings[key]
        c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        q = np.array([[c, -s], [s, c]])
        assert_allclose(case["rotation"], q, atol=0, rtol=0)
        roots = [np.roots([1, -(1 + beta - eta * eigenvalue), beta]) for eigenvalue in d]
        radius = max(float(np.max(np.abs(r))) for r in roots)
        assert_allclose(case["spectral_radius"], radius, atol=1e-7, rtol=1e-7)
        assert case["strictly_stable"] == bool(eta < 2 * (1 + beta) / largest)
        assert len(case["runs"]) == 2
        for index, record in enumerate(case["runs"]):
            assert record["orientation"] == ("diagonal" if index == 0 else "rotated")
            basis = np.eye(2) if index == 0 else q
            hessian = (basis * d) @ basis.T
            previous = current = z0.copy()
            expected = [basis @ z0]
            first_attainment = None
            initial_loss = 0.5 * z0 @ (d * z0)
            expected_losses = [initial_loss]
            expected_gradients = [None]
            status = "finite_cap"
            for step in range(1, settings["max_steps"] + 1):
                # Independent modal position equation; no accumulated-gradient velocity.
                following = (1 + beta - eta * d) * current - beta * previous
                expected_gradients.append(float(np.linalg.norm(d * current)))
                previous, current = current, following
                expected.append(basis @ current)
                loss = float(0.5 * current @ (d * current))
                expected_losses.append(loss)
                if first_attainment is None and loss / initial_loss <= settings["tolerance"]:
                    first_attainment = step
                if loss / initial_loss > settings["explosive_threshold"]:
                    status = "explosive"
                    break
            expected = np.array(expected)
            actual = np.asarray(record["positions"])
            scale = max(1.0, np.linalg.norm(z0))
            assert_allclose(actual / scale, expected / scale, atol=atol, rtol=rtol)
            assert record["status"] == status
            assert record["gradient_calls"] == len(expected) - 1
            assert record["first_attainment"] == first_attainment
            assert record["threshold_retained"] == bool(
                status == "finite_cap" and expected_losses[-1] / initial_loss <= settings["tolerance"])
            assert_allclose(record["final_position"], actual[-1], atol=0, rtol=0)
            assert_allclose(record["initial_loss"], expected_losses[0], atol=atol, rtol=rtol)
            assert_allclose(record["final_loss"], expected_losses[-1], atol=atol, rtol=rtol)
            trace = record["trace"]
            assert [entry["step"] for entry in trace] == list(range(len(expected)))
            assert trace[0]["gradient_norm_before_step"] is None
            assert_allclose([entry["loss"] for entry in trace], expected_losses, atol=atol, rtol=rtol)
            assert_allclose([entry["relative_loss"] for entry in trace],
                            np.asarray(expected_losses) / initial_loss, atol=atol, rtol=rtol)
            assert_allclose([entry["gradient_norm_before_step"] for entry in trace[1:]],
                            expected_gradients[1:], atol=atol, rtol=rtol)
            # Recompute saved path losses directly in ambient coordinates as well.
            actual_losses = np.einsum("ti,ij,tj->t", actual, hessian, actual) / 2
            assert_allclose(actual_losses, expected_losses, atol=atol, rtol=rtol)
            metadata = record["metadata"]
            assert metadata["gradient_matrix_vector_products"] == len(expected) - 1
            assert metadata["loss_matrix_vector_products"] == len(expected)
            assert metadata["basis_transform_calls"] == 0
            err = np.max(np.linalg.norm(actual - expected, axis=1) /
                         np.maximum(scale, np.linalg.norm(expected, axis=1)))
            max_reconstruction_error = max(max_reconstruction_error, float(err))
        base, rotated = case["runs"]
        z, w = np.asarray(base["positions"]), np.asarray(rotated["positions"])
        position_error = float(np.max(np.linalg.norm(w - z @ q.T, axis=1)) / max(1, np.linalg.norm(z0)))
        losses = np.array([entry["loss"] for entry in base["trace"]])
        rotated_losses = np.array([entry["loss"] for entry in rotated["trace"]])
        loss_error = float(np.max(np.abs(rotated_losses - losses) / np.maximum(1, abs(losses))))
        assert_allclose(case["max_scaled_position_error"], position_error, atol=0, rtol=1e-12)
        assert_allclose(case["max_scaled_loss_error"], loss_error, atol=0, rtol=1e-12)
        for key in ("status", "gradient_calls", "first_attainment", "threshold_retained"):
            assert case[key] == rotated[key]
        assert_allclose(case["relative_final_loss"], rotated["final_loss"] / rotated["initial_loss"])
        for key, value in csv_row.items():
            assert value == ("" if case[key] is None else str(case[key]))
    assert summary["runs"] == 2 * len(schedule)
    for key in ("max_scaled_position_error", "max_scaled_loss_error"):
        assert summary[key] == max(case[key] for case in saved["cases"])
    root = Path(__file__).resolve().parents[2]
    for relative, expected_hash in summary["source_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected_hash, relative
    report = dict(status="passed", cases=len(schedule), trajectories=2 * len(schedule),
                  max_relative_reconstruction_error=max_reconstruction_error,
                  method="Independent modal position recurrence; seeded regeneration; loss, gradient, stopping and CSV checks",
                  source_hashes_checked=len(summary["source_sha256"]))
    marker.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/part1"))
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
