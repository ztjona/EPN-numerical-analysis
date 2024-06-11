from typing import Callable

import sympy as sym


def taylor_approx(*, fcn: Callable[[float], float], x0: float, n: int):
    """Approximate a function using the Taylor nth polynomial.
    ## Parameters
    ``fcn``: function to approximate
    ``x0``: point to approximate around
    ``n``: number of terms in the approximation
    ## Return
    ``taylor``: the Taylor nth polynomial
    """

    x = sym.symbols("x")

    f = sym.sympify(fcn(x))
    print(f)

    taylor: sym.Symbol = 0

    for i in range(n + 1):
        taylor += f.diff(x, i).subs(x, x0) / sym.factorial(i) * (x - x0) ** i

    return taylor


def lagrange_approx():
    """

    ## Parameters

    ``a``:

    ``b``:

    ## Return

    ``a``:

    """

    return
