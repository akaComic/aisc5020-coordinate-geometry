"""Part 1: GD / heavy-ball updates and spectral theory for SPD quadratics.

The heavy-ball velocity stores accumulated gradients (not displacements).
The common runner initializes it to zero, equivalently w[-1] = w[0].
"""

import numpy as np

from .contracts import OptimizerState, RunConfig


def gd_step(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> None:
    """One gradient call and one scalar step size in ambient coordinates."""
    state.position -= config.learning_rate * gradient
    state.step += 1


def heavy_ball_step(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> None:
    """Update velocity before position; do not apply Adam-style bias correction."""
    # v_(t+1) = beta v_t + gradient(w_t); w_(t+1) = w_t - eta v_(t+1).
    state.velocity = config.beta * state.velocity + gradient
    state.position -= config.learning_rate * state.velocity
    state.step += 1


def _spectrum(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("eigenvalues must be a nonempty finite positive vector")
    return values


def _parameters(learning_rate: float, beta: float = 0.0) -> None:
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(beta) or not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")


def gd_spectral_radius(eigenvalues: np.ndarray, learning_rate: float) -> float:
    """Worst coordinate contraction; the objective bound uses its square."""
    values = _spectrum(eigenvalues)
    _parameters(learning_rate)
    return float(np.max(np.abs(1 - learning_rate * values)))


def heavy_ball_spectral_radius(
    eigenvalues: np.ndarray, learning_rate: float, beta: float,
) -> float:
    """Largest root modulus of r²-(1+beta-eta*lambda)r+beta.

    This is an asymptotic state factor, not a per-step objective bound.
    Repeated roots can introduce a polynomial transient.
    """
    values = _spectrum(eigenvalues)
    _parameters(learning_rate, beta)
    a = 1 + beta - learning_rate * values
    discriminant = a * a - 4 * beta
    real_radius = (np.abs(a) + np.sqrt(np.maximum(discriminant, 0))) / 2
    return float(np.max(np.where(discriminant < 0, np.sqrt(beta), real_radius)))


def heavy_ball_is_stable(eigenvalues: np.ndarray, learning_rate: float, beta: float) -> bool:
    """Strict Schur stability for every mode, with 0 <= beta < 1.

    Equality at eta*L = 2*(1+beta) is marginal, not convergent.
    """
    values = _spectrum(eigenvalues)
    _parameters(learning_rate, beta)
    return bool(learning_rate < 2 * (1 + beta) / np.max(values))


def optimal_gd_parameters(eigenvalues: np.ndarray) -> tuple[float, float]:
    """Return constant eta and minimax state factor over [mu, L]."""
    values = _spectrum(eigenvalues)
    mu, largest = float(np.min(values)), float(np.max(values))
    return 2 / (largest + mu), (largest - mu) / (largest + mu)


def optimal_heavy_ball_parameters(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    """Return eta, beta, asymptotic minimax factor for an SPD quadratic."""
    values = _spectrum(eigenvalues)
    small, large = np.sqrt(np.min(values)), np.sqrt(np.max(values))
    rho = float((large - small) / (large + small))
    return float(4 / (large + small)**2), rho**2, rho
