"""Exact CPU reference model for three-qubit correlated-X decoding.

This module deliberately contains no tensor-network, plotting, or accelerator
code. It is the exhaustive correctness reference for later implementation
stages.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from math import fsum
from numbers import Integral

import numpy as np
from numpy.typing import NDArray


def _binary(value: int, owner: str) -> int:
    if not isinstance(value, Integral) or int(value) not in (0, 1):
        raise ValueError(f"{owner} must be a binary integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True, order=True)
class ErrorPattern:
    """Immutable three-qubit bit-flip error pattern."""

    e1: int
    e2: int
    e3: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "e1", _binary(self.e1, "e1"))
        object.__setattr__(self, "e2", _binary(self.e2, "e2"))
        object.__setattr__(self, "e3", _binary(self.e3, "e3"))

    @classmethod
    def from_value(cls, value: "ErrorPattern | tuple[int, int, int]") -> "ErrorPattern":
        """Validate and normalize an error-pattern-like value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("error pattern must be an ErrorPattern or a length-three tuple")
        return cls(*value)

    @property
    def bits(self) -> tuple[int, int, int]:
        return self.e1, self.e2, self.e3

    @property
    def weight(self) -> int:
        return self.e1 + self.e2 + self.e3

    def __iter__(self) -> Iterator[int]:
        return iter(self.bits)

    def __str__(self) -> str:
        return f"{self.e1}{self.e2}{self.e3}"


@dataclass(frozen=True, slots=True, order=True)
class Syndrome:
    """Immutable pair of repetition-code parity-check outcomes."""

    s1: int
    s2: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "s1", _binary(self.s1, "s1"))
        object.__setattr__(self, "s2", _binary(self.s2, "s2"))

    @classmethod
    def from_value(cls, value: "Syndrome | tuple[int, int]") -> "Syndrome":
        """Validate and normalize a syndrome-like value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("syndrome must be a Syndrome or a length-two tuple")
        return cls(*value)

    @property
    def bits(self) -> tuple[int, int]:
        return self.s1, self.s2

    def __iter__(self) -> Iterator[int]:
        return iter(self.bits)

    def __str__(self) -> str:
        return f"{self.s1}{self.s2}"


# The order follows the mapping in the example specification rather than
# integer or lexical order, making printed reference tables directly comparable.
ERROR_PATTERNS: tuple[ErrorPattern, ...] = tuple(
    ErrorPattern(*bits)
    for bits in (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
)


def syndrome_for_error(error: ErrorPattern | tuple[int, int, int]) -> Syndrome:
    """Return ``(e1 XOR e2, e2 XOR e3)``."""
    pattern = ErrorPattern.from_value(error)
    return Syndrome(pattern.e1 ^ pattern.e2, pattern.e2 ^ pattern.e3)


_RECOVERY_TABLE = {
    Syndrome(0, 0): ErrorPattern(0, 0, 0),
    Syndrome(1, 0): ErrorPattern(1, 0, 0),
    Syndrome(1, 1): ErrorPattern(0, 1, 0),
    Syndrome(0, 1): ErrorPattern(0, 0, 1),
}


def majority_vote_recovery(
    syndrome: Syndrome | tuple[int, int],
) -> ErrorPattern:
    """Return the fixed minimum-weight recovery for a measured syndrome."""
    return _RECOVERY_TABLE[Syndrome.from_value(syndrome)]


def residual_after_recovery(
    error: ErrorPattern | tuple[int, int, int],
    recovery: ErrorPattern | tuple[int, int, int] | None = None,
) -> ErrorPattern:
    """Apply a recovery by XOR and return the residual physical operator."""
    pattern = ErrorPattern.from_value(error)
    correction = (
        majority_vote_recovery(syndrome_for_error(pattern))
        if recovery is None
        else ErrorPattern.from_value(recovery)
    )
    return ErrorPattern(*(left ^ right for left, right in zip(pattern, correction)))


def logical_failure(error: ErrorPattern | tuple[int, int, int]) -> bool:
    """Determine failure from the post-recovery residual, not error weight."""
    residual = residual_after_recovery(error)
    if residual == ErrorPattern(0, 0, 0):
        return False
    if residual == ErrorPattern(1, 1, 1):
        return True
    raise RuntimeError(
        f"majority-vote recovery produced invalid code-space residual {residual}"
    )


@dataclass(frozen=True, slots=True)
class CorrelatedNoiseDiagnostics:
    """Exact physical diagnostics derived from all eight patterns."""

    physical_error_rate: float
    zero_error_probability: float
    exactly_one_error_probability: float
    two_or_more_error_probability: float
    xxx_probability: float
    average_physical_error_weight: float
    weight_probabilities: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CorrelatedXXXNoise:
    """Shared-latent correlated XXX noise with independent local X flips.

    ``c ~ Bernoulli(rho)``, ``u_i ~ Bernoulli(p)``, and ``e_i = c XOR u_i``.
    Thus ``p`` is the local latent-flip probability, not the final physical
    error rate when ``rho`` is nonzero.
    """

    p: float
    rho: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must satisfy 0 <= p <= 1, got {self.p!r}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must satisfy 0 <= rho <= 1, got {self.rho!r}")

    def error_probability(
        self,
        error_pattern: ErrorPattern | tuple[int, int, int],
    ) -> float:
        """Compute an error-pattern probability by summing both latent sectors."""
        error = ErrorPattern.from_value(error_pattern)
        total = 0.0
        for correlated, correlated_probability in (
            (0, 1.0 - self.rho),
            (1, self.rho),
        ):
            local_bits = tuple(bit ^ correlated for bit in error)
            local_probability = 1.0
            for bit in local_bits:
                local_probability *= self.p if bit else 1.0 - self.p
            total += correlated_probability * local_probability
        return float(total)

    def sample_error(self, rng: np.random.Generator) -> ErrorPattern:
        """Draw one pattern from a caller-owned NumPy random generator."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        correlated = int(rng.random() < self.rho)
        local = rng.random(3) < self.p
        return ErrorPattern(*(correlated ^ int(bit) for bit in local))

    def sample_batch(
        self,
        num_shots: int,
        seed: int | None = None,
    ) -> NDArray[np.int8]:
        """Vectorize latent and local sampling into a ``(num_shots, 3)`` array."""
        if not isinstance(num_shots, Integral) or isinstance(num_shots, bool):
            raise TypeError("num_shots must be an integer")
        if num_shots < 0:
            raise ValueError("num_shots must be nonnegative")
        rng = np.random.default_rng(seed)
        correlated = (rng.random((int(num_shots), 1)) < self.rho).astype(np.int8)
        local = (rng.random((int(num_shots), 3)) < self.p).astype(np.int8)
        return np.bitwise_xor(correlated, local).astype(np.int8, copy=False)

    def probability_table(self) -> tuple[tuple[ErrorPattern, float], ...]:
        """Return the deterministic exact probability table."""
        return tuple((error, self.error_probability(error)) for error in ERROR_PATTERNS)

    def physical_error_rate(self) -> float:
        """Return the average final error probability per physical qubit."""
        return float(self.p + self.rho - 2.0 * self.p * self.rho)

    def error_weight_probabilities(self) -> tuple[float, float, float, float]:
        """Return exact probabilities for physical weights zero through three."""
        return tuple(
            fsum(probability for error, probability in self.probability_table() if error.weight == weight)
            for weight in range(4)
        )

    def diagnostics(self) -> CorrelatedNoiseDiagnostics:
        """Calculate exact physical-rate and error-weight diagnostics."""
        weights = self.error_weight_probabilities()
        return CorrelatedNoiseDiagnostics(
            physical_error_rate=self.physical_error_rate(),
            zero_error_probability=weights[0],
            exactly_one_error_probability=weights[1],
            two_or_more_error_probability=weights[2] + weights[3],
            xxx_probability=self.error_probability(ErrorPattern(1, 1, 1)),
            average_physical_error_weight=fsum(
                weight * probability for weight, probability in enumerate(weights)
            ),
            weight_probabilities=weights,
        )


class LogicalClass(str, Enum):
    """Residual logical class after fixed majority-vote recovery."""

    I_L = "I_L"
    X_L = "X_L"


class LogicalErrorType(str, Enum):
    """Logical error reported by the exhaustive decoder."""

    NONE = "none"
    LOGICAL_X = "logical_x"


@dataclass(frozen=True, slots=True)
class CandidateDiagnostic:
    """One syndrome-consistent physical error and its decoded residual."""

    error: ErrorPattern
    probability: float
    recovery: ErrorPattern
    residual: ErrorPattern
    logical_class: LogicalClass
    logical_error: bool


@dataclass(frozen=True, slots=True)
class DecodeDiagnostics:
    """Inspectable exhaustive details for one syndrome."""

    syndrome_probability: float
    candidates: tuple[CandidateDiagnostic, ...]
    logical_joint_probabilities: tuple[float, float]
    logical_posteriors: tuple[float, float]
    conditional_probability_defined: bool


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Result of exhaustive syndrome-conditioned reference decoding."""

    syndrome: Syndrome
    selected_recovery: ErrorPattern
    candidate_error_probabilities: tuple[tuple[ErrorPattern, float], ...]
    logical_error_flag: bool
    logical_error_type: LogicalErrorType
    selected_logical_class: LogicalClass
    posterior_probability: float
    diagnostic_information: DecodeDiagnostics


def decode_syndrome(
    syndrome: Syndrome | tuple[int, int],
    noise_model: CorrelatedXXXNoise,
) -> DecodeResult:
    """Enumerate all errors and decode one syndrome exactly.

    The selected logical class is the most probable residual class under the
    fixed recovery. Ties, including impossible syndromes, select ``I_L``.
    """
    if not isinstance(noise_model, CorrelatedXXXNoise):
        raise TypeError("noise_model must be CorrelatedXXXNoise")
    measured = Syndrome.from_value(syndrome)
    recovery = majority_vote_recovery(measured)
    candidates: list[CandidateDiagnostic] = []
    logical_joint = [0.0, 0.0]
    for error in ERROR_PATTERNS:
        if syndrome_for_error(error) != measured:
            continue
        probability = noise_model.error_probability(error)
        residual = residual_after_recovery(error, recovery)
        failed = residual == ErrorPattern(1, 1, 1)
        logical_class = LogicalClass.X_L if failed else LogicalClass.I_L
        logical_joint[int(failed)] += probability
        candidates.append(
            CandidateDiagnostic(
                error,
                probability,
                recovery,
                residual,
                logical_class,
                failed,
            )
        )
    syndrome_probability = fsum(logical_joint)
    conditional_defined = syndrome_probability > 0.0
    posteriors = (
        tuple(value / syndrome_probability for value in logical_joint)
        if conditional_defined
        else (0.0, 0.0)
    )
    selected_index = int(logical_joint[1] > logical_joint[0])
    selected_class = LogicalClass.X_L if selected_index else LogicalClass.I_L
    return DecodeResult(
        syndrome=measured,
        selected_recovery=recovery,
        candidate_error_probabilities=tuple(
            (candidate.error, candidate.probability) for candidate in candidates
        ),
        logical_error_flag=bool(selected_index),
        logical_error_type=(
            LogicalErrorType.LOGICAL_X if selected_index else LogicalErrorType.NONE
        ),
        selected_logical_class=selected_class,
        posterior_probability=posteriors[selected_index],
        diagnostic_information=DecodeDiagnostics(
            syndrome_probability,
            tuple(candidates),
            (logical_joint[0], logical_joint[1]),
            (posteriors[0], posteriors[1]),
            conditional_defined,
        ),
    )


@dataclass(frozen=True, slots=True)
class ExactLogicalDiagnostics:
    """Exact physical, syndrome, and post-recovery logical probabilities."""

    physical_error_rate: float
    logical_error_rate: float
    error_weight_probabilities: tuple[float, float, float, float]
    syndrome_probabilities: tuple[tuple[Syndrome, float], ...]
    logical_coset_probabilities: tuple[tuple[Syndrome, tuple[float, float]], ...]


def exact_logical_error_rate(noise_model: CorrelatedXXXNoise) -> float:
    """Sum probabilities of patterns whose actual residual is logical X."""
    if not isinstance(noise_model, CorrelatedXXXNoise):
        raise TypeError("noise_model must be CorrelatedXXXNoise")
    return fsum(
        noise_model.error_probability(error)
        for error in ERROR_PATTERNS
        if logical_failure(error)
    )


def exact_logical_diagnostics(
    noise_model: CorrelatedXXXNoise,
) -> ExactLogicalDiagnostics:
    """Enumerate all eight patterns and aggregate exact decoding diagnostics."""
    if not isinstance(noise_model, CorrelatedXXXNoise):
        raise TypeError("noise_model must be CorrelatedXXXNoise")
    syndromes = (Syndrome(0, 0), Syndrome(0, 1), Syndrome(1, 0), Syndrome(1, 1))
    decoded = tuple(decode_syndrome(syndrome, noise_model) for syndrome in syndromes)
    return ExactLogicalDiagnostics(
        physical_error_rate=noise_model.physical_error_rate(),
        logical_error_rate=exact_logical_error_rate(noise_model),
        error_weight_probabilities=noise_model.error_weight_probabilities(),
        syndrome_probabilities=tuple(
            (result.syndrome, result.diagnostic_information.syndrome_probability)
            for result in decoded
        ),
        logical_coset_probabilities=tuple(
            (result.syndrome, result.diagnostic_information.logical_joint_probabilities)
            for result in decoded
        ),
    )
