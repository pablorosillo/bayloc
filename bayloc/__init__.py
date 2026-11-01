"""bayloc: memory-decay imputation of missing locations in mobility timelines.

Reference implementation of Rosillo-Rodes, Tugores, Sanchez and Ramasco,
"Inferring missing locations in human mobility data using memory effects".
"""
from .core import (BASELINES, KERNELS, LAMBDA_INF, BayLoc,
                   InsufficientDataError, LambdaOptimizationError,
                   build_lambda_grid, fit_lambda, predict, score,
                   three_way_split)

__version__ = "2.0.0"

__all__ = ["BayLoc", "InsufficientDataError", "LambdaOptimizationError",
           "build_lambda_grid", "fit_lambda", "predict", "score",
           "three_way_split", "BASELINES", "KERNELS", "LAMBDA_INF",
           "__version__"]
