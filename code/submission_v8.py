"""Inference entry for the separately packaged V8 development candidate.

The caller supplies bank, arrays, and checkpoint. No split is loaded here.
Set BLAS/OpenMP environment limits before starting Python for four-thread runs.
"""
from v8_stack_model import predict

__all__ = ['predict']
