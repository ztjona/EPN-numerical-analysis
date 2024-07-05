# -*- coding: utf-8 -*-
"""
Python 3
05 / 07 / 2024
@author: z_tjona

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth
"""

# ----------------------------- logging --------------------------
import logging
from sys import stdout
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    stream=stdout,
    datefmt="%m-%d %H:%M:%S",
)
logging.info(datetime.now())

import numpy as np


# ####################################################################
def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de eliminación gaussiana.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(
            A, dtype=float
        )  # convertir en float, porque si no, convierte en enteros
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]
    n_adds = 0
    n_mults = 0
    for i in range(0, n - 1):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            logging.info(f"\n{A}")
            raise ValueError("No existe solución única.")

        if p != i:
            logging.info(f"Intercambiando filas {i} y {p}.")
            # swap rows
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]
            n_mults += 1 + (n + 1 - i - 1)
            n_adds += n + 1 - i - 1

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        # Sin embargo, esto solo se accede al finalizar la matriz... Con todos los pivotes
        if A[n - 1, n] == 0:
            raise ValueError("Infinitas soluciones.")
        else:
            raise ValueError("Sin solución.")

    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]

    n_mults += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        n_adds -= 1
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
            n_mults += 1
            n_adds += 1
        solucion[i] = (A[i, n] - suma) / A[i, i]
        n_adds += 1
        n_mults += 1

    return solucion, n_adds, n_mults


# ####################################################################
def gauss_jordan(A: np.ndarray) -> np.ndarray:
    """Resuelve un sistema de ecuaciones lineales mediante el método de Gauss-Jordan.

    ## Parameters

    ``A``: matriz aumentada del sistema de ecuaciones lineales. Debe ser de tamaño n-by-(n+1), donde n es el número de incógnitas.

    ## Return

    ``solucion``: vector con la solución del sistema de ecuaciones lineales.

    """
    if not isinstance(A, np.ndarray):
        logging.debug("Convirtiendo A a numpy array.")
        A = np.array(
            A, dtype=float
        )  # convertir en float, porque si no, convierte en enteros
    assert A.shape[0] == A.shape[1] - 1, "La matriz A debe ser de tamaño n-by-(n+1)."
    n = A.shape[0]
    n_adds = 0
    n_mults = 0
    for i in range(0, n):  # loop por columna

        # --- encontrar pivote
        p = None  # default, first element
        for pi in range(i, n):
            if A[pi, i] == 0:
                # must be nonzero
                continue

            if p is None:
                # first nonzero element
                p = pi
                continue

            if abs(A[pi, i]) < abs(A[p, i]):
                p = pi

        if p is None:
            # no pivot found.
            logging.info(f"\n{A}")
            raise ValueError("No existe solución única.")

        if p != i:
            logging.info(f"Intercambiando filas {i} y {p}.")
            # swap rows
            _aux = A[i, :].copy()
            A[i, :] = A[p, :].copy()
            A[p, :] = _aux

        # --- Eliminación: loop por fila
        for j in range(n):
            if i == j:
                continue
            m = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - m * A[i, i:]
            n_mults += 1 + (n + 1 - i - 1)
            n_adds += n + 1 - i - 1

        logging.info(f"\n{A}")

    if A[n - 1, n - 1] == 0:
        # Sin embargo, esto solo se accede al finalizar la matriz... Con todos los pivotes
        if A[n - 1, n] == 0:
            raise ValueError("Infinitas soluciones.")
        else:
            raise ValueError("Sin solución.")

    # --- Sustitución hacia atrás
    solucion = np.zeros(n)
    for i in range(n):
        solucion[i] = A[i, n] / A[i, i]
        n_mults += 1

    return solucion, n_adds, n_mults
