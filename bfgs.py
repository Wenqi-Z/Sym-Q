"""
This module provides an implementation of the BFGS optimization algorithm, specifically
tailored for our use case. It is a modified version of the implementation found in 
Joint_Supervised_Learning_for_SR/src/architectures/bfgs.py
"""

import time

import numpy as np
import sympy as sp
import torch
from scipy.optimize import minimize


import re
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Custom module dictionary
MODULES = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # Define 'coth' using np.cosh and np.sinh as NumPy doesn't have a direct coth function
    "coth": lambda x: np.cosh(x) / np.sinh(x),
    "sqrt": np.sqrt,
    "log": np.log,
    "exp": np.exp,
    "Abs": np.abs,
    "numpy": np,  # Include numpy for other functions and operations
}


class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def bfgs(pred_str, X, y):
    idx_remove = True
    total_variables = ["x_1", "x_2"]

    # Check where dimensions not use, and replace them with 1 to avoid numerical issues with BFGS (i.e. absent variables placed in the denominator)
    y = y.squeeze()
    X = X.clone()
    bool_dim = (X == 0).all(axis=1).squeeze()
    X[:, :, bool_dim] = 1

    candidate = re.sub(r"\bc\b", "constant", pred_str)

    expr = candidate
    for i in range(candidate.count("constant")):
        expr = expr.replace("constant", f"c{i}", 1)

    # print('Constructing BFGS loss...')

    # if cfg.bfgs.idx_remove:
    if idx_remove:
        # print('Flag idx remove ON, Removing indeces with high values...')
        bool_con = (X < 200).all(axis=2).squeeze()
        X = X[:, bool_con, :]

    max_y = np.max(np.abs(torch.abs(y).cpu().numpy()))
    # print('checking input values range...')
    # if max_y > 300:
    #     print('Attention, input values are very large. Optimization may fail due to numerical issues')

    diffs = []
    for i in range(X.shape[1]):
        curr_expr = expr
        # for idx, j in enumerate(cfg.total_variables):
        for idx, j in enumerate(total_variables):
            curr_expr = sp.sympify(curr_expr).subs(j, X[:, i, idx])
        diff = curr_expr - y[i]
        diffs.append(diff)

    loss = np.mean(np.square(diffs))

    # Lists where all restarted will be appended
    F_loss = []
    consts_ = []
    funcs = []
    symbols = {i: sp.Symbol(f"c{i}") for i in range(candidate.count("constant"))}

    # Ensure 20 valid trys
    num = 0
    tryout = 0
    while num < 20 and tryout < 50:
        # Compute number of coefficients
        np.random.seed(tryout)
        x0 = np.random.randn(len(symbols))

        s = list(symbols.values())
        # bfgs optimization
        fun_timed = TimedFun(fun=sp.lambdify(s, loss, modules=["numpy"]))
        if len(x0):
            try:
                minimize(
                    fun_timed.fun, x0, method="BFGS"
                )  # check consts interval and if they are int
            except Exception as e:
                print(f"Encountered in bfgs: {e}")
                tryout += 1
                continue
            consts_.append(fun_timed.x)
        else:
            consts_.append([])

        final = expr
        for i in range(len(s)):
            final = sp.sympify(final).replace(s[i], fun_timed.x[i])

        funcs.append(final)

        values = {x: X[:, :, idx].cpu() for idx, x in enumerate(total_variables)}

        # Use the custom module dictionary in lambdify
        y_found = sp.lambdify(",".join(total_variables), final, modules=MODULES)(
            **values
        )
        final_loss = np.mean(np.square(y_found - y.cpu()).numpy())
        F_loss.append(final_loss)

        if not np.isnan(final_loss):
            num += 1
        tryout += 1

    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        k_best = 0

    # guard against domain problem
    no_domain = False
    if np.isnan(F_loss[k_best]):
        no_domain = True

    return funcs[k_best], consts_[k_best], F_loss[k_best], expr, no_domain
