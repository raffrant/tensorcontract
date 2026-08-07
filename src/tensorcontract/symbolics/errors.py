"""Validation errors for the minimal symbolic intermediate representation."""


class SymbolicValidationError(ValueError):
    """Base class for invalid symbolic graphs."""


class MissingIndexError(SymbolicValidationError):
    """Raised when a tensor or operation references an unknown index."""


class MissingValueError(SymbolicValidationError):
    """Raised when an operation references an unknown tensor or operation."""


class DimensionMismatchError(SymbolicValidationError):
    """Raised when a declared tensor shape conflicts with index dimensions."""


class RepeatedIndexError(SymbolicValidationError):
    """Raised when an index is repeated where diagonal semantics are not explicit."""


class InvalidOperationError(SymbolicValidationError):
    """Raised when operation indices or dependencies are inconsistent."""
