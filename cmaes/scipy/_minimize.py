from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import OptimizeResult, Bounds

from cmaes import CMA, SepCMA, CMAwM

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable
    from typing import Literal
    from typing import Optional
    from typing import Sequence
    from typing import Union
    from mypy_extensions import VarArg


_EPS = 1e-10


def minimize(
    fun: Callable[[np.ndarray, VarArg(Any)], float],
    x0: np.ndarray,
    bounds: Union[Sequence[Bounds], Sequence[tuple[float, float]]],
    *,
    args: Optional[tuple[Any, ...]] = None,
    method: Literal["cma", "sep-cma"] = "cma",
    tol: Optional[float],
) -> OptimizeResult:
    """Minimization of scalar function of one or more variables like
    `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_

    Example:

        .. code::

           import numpy as np
           from cmaes import minimize

           def quadratic(x1, x2):
               return (x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2

           optimizer = CMA(mean=np.zeros(2), sigma=1.3)

           for generation in range(50):
               solutions = []
               for _ in range(optimizer.population_size):
                   # Ask a parameter
                   x = optimizer.ask()
                   value = quadratic(x[0], x[1])
                   solutions.append((x, value))
                   print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")

               # Tell evaluation values.
               optimizer.tell(solutions)

    Args:
        fun:
            The objective function to be minimized. ``fun(x, *args) -> float``.

        x0:
            Initial guess. A ndarray of real elements of size (n,), where ``n`` is the number of independent variables.

        method:
            Type of solver. Must be either "cma" or "sep-cma".

        args:
            Extra arguments passed to the objective function (optional).

        bounds:
            Bounds on variable. There are two ways to specify the bounds. Instance of
            `Bounds <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.Bounds.html#scipy.optimize.Bounds>`_
            class, or Sequence of ``(min, max)`` pairs for each element in ``x``.

        tol:
            Tolerance for termination (optional).

    """
    # TODO(c-bata): Support callback argument.
    assert len(x0) > 1, "The dimension of x0 must be larger than 1"
    assert len(x0) == len(bounds), "The dimensions of x0 and bounds must be same"

    sigma = None
    cma_bounds = []
    for b in bounds:

        if isinstance(b, tuple):
            assert len(b) == 2
            assert b[1] > b[0], "upper bounds must be larger than lower bounds"
            cma_bounds.append(b)
        elif isinstance(b, Bounds):
            cma_bounds.append((b.lb, b.ub))
        else:
            assert False, (
                "bounds must be either sequence of (min, max) "
                "or a sequence of scipy.optimize.Bounds instance."
            )
        sigma = min(sigma, (cma_bounds[-1][1] - cma_bounds[-1][0]) / 6)
    sigma = max(sigma, _EPS)

    if method == "cma":
        optimizer = CMA(
            mean=x0,
            sigma=sigma,
            bounds=np.ndarray(bounds),
        )
    elif method == "sep-cma":
        optimizer = SepCMA(
            mean=x0,
            sigma=sigma,
            bounds=np.ndarray(bounds),
        )
    else:
        assert False, "Unsupported method"

    while True:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = fun(x, *args)
            solutions.append((x, value))
        optimizer.tell(solutions)



        if optimizer.should_stop():
            break

    result: Optional[OptimizeResult] = None
