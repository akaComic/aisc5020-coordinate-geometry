"""Module 1: fixed-parameter theory checks, not a tuned main experiment."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
from numpy.testing import assert_allclose

from .contracts import RunConfig
from .gd_momentum import (
    gd_spectral_radius, heavy_ball_is_stable, heavy_ball_spectral_radius,
    optimal_gd_parameters, optimal_heavy_ball_parameters,
)
from .problems import QuadraticProblem, matched_pair, rotation_2d
from .runner import run


def candidates(values):
    """Include convergent runs, exact stability boundaries and unstable controls."""
    largest = float(np.max(values))
    gd_rate, _ = optimal_gd_parameters(values)
    hb_rate, beta, _ = optimal_heavy_ball_parameters(values)
    return [
        ("gd_inverse_L", "gd", 1 / largest, 0.0),
        ("gd_optimal", "gd", gd_rate, 0.0),
        ("gd_boundary", "gd", 2 / largest, 0.0),
        ("gd_unstable", "gd", 2.1 / largest, 0.0),
        ("hb_optimal", "heavy_ball", hb_rate, beta),
        ("hb_boundary", "heavy_ball", 2 * (1 + beta) / largest, beta),
        ("hb_unstable", "heavy_ball", 2.1 * (1 + beta) / largest, beta),
    ]


class _RecordedProblem:
    """Observe the public runner's gradient inputs without adding gradient calls."""

    def __init__(self, problem):
        self.problem = problem
        self.positions = []

    @property
    def dimension(self):
        return self.problem.dimension

    def gradient(self, position):
        self.positions.append(position.copy())
        return self.problem.gradient(position)

    def loss(self, position):
        return self.problem.loss(position)


def recorded_run(problem, initial, config):
    observed = _RecordedProblem(problem)
    result = run(observed, initial, config)
    # These controls stop while finite, so the last output is the actual last step.
    if result.status == "nonfinite":
        raise AssertionError("Unexpected nonfinite result; reduce the control budget")
    path = np.vstack([*observed.positions, result.final_position])
    return result, path


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes(root):
    paths = sorted((root / "src/coordinate_geometry").glob("*.py"))
    paths += [root / "configs/part1.json"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/part1.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part1"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings["study_status"] != "part1_theory_validation":
        parser.error("Expected part1_theory_validation configuration")
    values = np.asarray(settings["eigenvalues"], dtype=float)
    if values.shape != (2,):
        parser.error("The configured angle study requires exactly two eigenvalues")
    canonical, _, z0, _ = matched_pair(values, np.eye(2), np.random.default_rng(settings["seed"]))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "artifact_verification.json").unlink(missing_ok=True)
    atol, rtol = settings["verification_atol"], settings["verification_rtol"]
    scale = max(1.0, float(np.linalg.norm(z0)))
    cases, rows = [], []
    for angle in settings["angles_degrees"]:
        q = rotation_2d(np.deg2rad(angle))
        rotated = QuadraticProblem(values, q)
        for name, algorithm, rate, beta in candidates(values):
            config = RunConfig(algorithm, rate, settings["max_steps"], beta=beta,
                               tolerance=settings["tolerance"],
                               explosive_threshold=settings["explosive_threshold"])
            base, z = recorded_run(canonical, z0, config)
            other, w = recorded_run(rotated, q @ z0, config)
            assert base.status == other.status
            assert base.gradient_calls == other.gradient_calls
            assert base.first_attainment == other.first_attainment
            assert_allclose(w / scale, (z @ q.T) / scale, atol=atol, rtol=rtol)
            base_loss = np.array([entry.loss for entry in base.trace])
            other_loss = np.array([entry.loss for entry in other.trace])
            assert_allclose(other_loss, base_loss, atol=atol, rtol=rtol)
            radius = (gd_spectral_radius(values, rate) if algorithm == "gd" else
                      heavy_ball_spectral_radius(values, rate, beta))
            stable = (rate < 2 / max(values) if algorithm == "gd" else
                      heavy_ball_is_stable(values, rate, beta))
            error = float(np.max(np.linalg.norm(w - z @ q.T, axis=1)) / scale)
            loss_error = float(np.max(np.abs(other_loss - base_loss) / np.maximum(1, np.abs(base_loss))))
            row = dict(case_id=f"{name}__angle_{angle}", candidate=name,
                       angle_degrees=angle, algorithm=algorithm, learning_rate=rate,
                       beta=beta, spectral_radius=radius, strictly_stable=bool(stable),
                       status=other.status, gradient_calls=other.gradient_calls,
                       first_attainment=other.first_attainment,
                       threshold_retained=other.threshold_retained,
                       relative_final_loss=other.final_loss / other.initial_loss,
                       max_scaled_position_error=error, max_scaled_loss_error=loss_error)
            rows.append(row)
            saved_runs = []
            for orientation, result, path in (("diagonal", base, z), ("rotated", other, w)):
                saved_runs.append(dict(orientation=orientation, positions=path.tolist(),
                                       trace=[asdict(entry) for entry in result.trace],
                                       final_position=result.final_position.tolist(),
                                       initial_loss=result.initial_loss, final_loss=result.final_loss,
                                       status=result.status, gradient_calls=result.gradient_calls,
                                       first_attainment=result.first_attainment,
                                       threshold_retained=result.threshold_retained,
                                       metadata=result.metadata))
            cases.append(dict(**row, config=asdict(config), rotation=q.tolist(), runs=saved_runs))
    root = Path(__file__).resolve().parents[2]
    (args.output / "run_config.json").write_bytes(args.config.read_bytes())
    _write_json(args.output / "trajectories.json", dict(eigenvalues=values.tolist(),
                canonical_initial=z0.tolist(), cases=cases))
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(study_status=settings["study_status"], protocol_frozen=False,
                   generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   python=platform.python_version(), numpy=np.__version__,
                   code_version="source SHA256 manifest; this folder has no required Git checkout",
                   source_sha256=source_hashes(root),
                   config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                   cases=len(cases), runs=2 * len(cases),
                   max_scaled_position_error=max(row["max_scaled_position_error"] for row in rows),
                   max_scaled_loss_error=max(row["max_scaled_loss_error"] for row in rows),
                   scope="Fixed theory parameters and stability controls; no tuning or statistical claims")
    _write_json(args.output / "summary.json", summary)
    if not args.no_plots:
        plot_cases(cases, args.output, settings["tolerance"])
    print(json.dumps({key: summary[key] for key in
                      ("cases", "runs", "max_scaled_position_error", "max_scaled_loss_error")}, indent=2))
    return 0


def plot_cases(cases, output, tolerance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"gd_inverse_L": "tab:blue", "gd_optimal": "tab:orange", "hb_optimal": "tab:green"}
    angles = list(dict.fromkeys(case["angle_degrees"] for case in cases))
    styles = ["-", "--", "-.", ":"]
    for case in cases:
        if case["candidate"] in ("gd_inverse_L", "gd_optimal", "hb_optimal"):
            trace = case["runs"][1]["trace"]
            axes[0].semilogy([t["step"] for t in trace],
                             [max(t["relative_loss"], np.finfo(float).tiny) for t in trace],
                             color=colors[case["candidate"]],
                             linestyle=styles[angles.index(case["angle_degrees"]) % len(styles)],
                             label=f"{case['candidate']}, {case['angle_degrees']} deg")
    axes[0].axhline(tolerance, color="gray", linestyle=":", label=f"target {tolerance:g}")
    axes[0].set(xlabel="Gradient evaluations", ylabel="Relative objective gap",
                title="Stable theory parameters; paired rotations")
    axes[0].legend(fontsize=6, ncol=2)
    first_angle = cases[0]["angle_degrees"]
    for case in cases:
        if case["angle_degrees"] == first_angle and case["candidate"] in (
                "gd_boundary", "gd_unstable", "hb_boundary", "hb_unstable"):
            trace = case["runs"][1]["trace"]
            axes[1].semilogy([t["step"] for t in trace], [t["relative_loss"] for t in trace],
                             label=f"{case['candidate']} ({case['status']})")
    axes[1].set(xlabel="Gradient evaluations", ylabel="Relative objective gap",
                title="Boundary and unstable controls (diagonal case)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "part1_validation.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
