import os
import re
import signal
import argparse
import warnings
import hydra
import h5py
import psutil
import random
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import sympy
from sympy import sympify, Float, Symbol
from sympy.core.rules import Transform
from torch.distributions.uniform import Uniform

from nesymres.utils import load_eq, load_metadata_hdf5
from nesymres.dataset.generator import Generator, UnknownSymPyOperator

from utils import handle_timeout, load_cfg, get_seq2action


def constants_to_placeholder(s, symbol="c"):
    sympy_expr = sympify(s)
    sympy_expr = sympy_expr.xreplace(
        Transform(
            lambda x: Symbol(symbol, real=True, nonzero=True),
            lambda x: isinstance(x, Float),
        )
    )
    return sympy_expr


def sample_symbolic_constants_from_coeff_dict(coeff_dict, cfg=None):
    dummy_consts = {const: 1 if const[:2] == "cm" else 0 for const in coeff_dict.keys()}
    consts = dummy_consts.copy()
    if cfg:
        used_consts = random.randint(0, min(len(coeff_dict), cfg.num_constants))
        symbols_used = random.sample(set(coeff_dict.keys()), used_consts)
        for si in symbols_used:
            if si[:2] == "ca":
                consts[si] = round(
                    float(Uniform(cfg.additive.min, cfg.additive.max).sample()), 3
                )
            elif si[:2] == "cm":
                consts[si] = round(
                    float(
                        Uniform(cfg.multiplicative.min, cfg.multiplicative.max).sample()
                    ),
                    3,
                )
            else:
                raise KeyError
    else:
        consts = dummy_consts
    return consts, dummy_consts


def filter_valid_points(X, Y):
    valid_indices = (
        np.isfinite(Y) & (~np.iscomplex(Y)) & (~np.isnan(Y)) & (abs(Y) < 5e4)
    )
    X_valid = X[:, valid_indices]
    Y_valid = Y[valid_indices]
    return X_valid, Y_valid


def generateDataFast(
    eq, n_points, n_vars, decimals, min_x, max_x
):
    total_variables = [f"x_{i}" for i in range(1, n_vars + 1)]
    symbols = sympy.symbols(total_variables)
    expr = sympy.sympify(eq)
    lambdified_expr = sympy.lambdify(symbols, expr, "numpy")

    initial_n_points = n_points * 30  # Generate 30 times more points initially
    X = np.round(
        np.random.uniform(min_x, max_x, (n_vars, initial_n_points)),
        decimals=decimals,
    )

    # Set the values of variables that are not present in the expression to zero
    for i in range(n_vars):
        if total_variables[i] not in eq:
            X[i, :] = 0.0

    # Evaluate the expression for all generated values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Y = lambdified_expr(*X)

    if isinstance(Y, float) or isinstance(Y, int):
        return None

    # Filter out invalid values (NaN, Inf, or complex numbers)
    X_valid, Y_valid = filter_valid_points(X, Y)

    # Check if the number of valid points is less than the required number
    num_valid_points = len(Y_valid)
    if num_valid_points < n_points:
        return None
    else:
        # Select the required subset of valid points
        X = X_valid[:, :n_points]
        Y = Y_valid[:n_points]

    return np.vstack([X, np.expand_dims(Y, axis=0)])


def process_equation(eq, keys, eq_id, constants_cfg):
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, 0.4)

    warnings.filterwarnings("ignore")

    try:
        eq_no_consts = eq.expr
        consts_elemns = eq.coeff_dict
        # print(f"eq: {eq}")
        # print(f"eq_no_consts: {eq_no_consts}")
        # print(f"consts_elemns: {consts_elemns}")

        w_const, wout_consts = sample_symbolic_constants_from_coeff_dict(
            consts_elemns, constants_cfg
        )
        eq_string = eq_no_consts.format(**w_const)
        eq_infix = str(sympy.sympify(eq_string)).replace(" ", "")
        # print(f"w_const: {w_const}")
        # print(f"wout_consts: {wout_consts}")
        # print(f"eq_string: {eq_string}")
        # print(f"eq_infix: {eq_infix}")

        if (
            "zoo" in eq_infix
            or "nan" in eq_infix
            or "I" in eq_infix
            or "E" in eq_infix
            or "pi" in eq_infix
        ):
            return None

        exps = re.findall(r"(\*\*[0-9\.]+)", eq_infix)
        # print(f"exps: {exps}")
        for ex in exps:
            cexp = "**" + str(
                eval(ex[2:]) if eval(ex[2:]) < 6 else np.random.randint(2, 6)
            )
            eq_infix = eq_infix.replace(ex, cexp)
        # print(f"eq_infix: {eq_infix}")

        try:
            eq_sympy_infix = constants_to_placeholder(eq_infix)
            eq_skeleton = str(eq_sympy_infix).replace(" ", "")
            traversal = Generator.sympy_to_prefix(eq_sympy_infix)
        except (UnknownSymPyOperator, RecursionError):
            del (
                eq_no_consts,
                consts_elemns,
                w_const,
                wout_consts,
                eq_string,
                eq_infix,
                exps,
            )
            return None

        # print(f"eq_sympy_infix: {eq_sympy_infix}")
        # print(f"eq_skeleton: {eq_skeleton}")
        # print(f"traversal: {traversal}")

        if any(val not in keys for val in traversal):
            return None

        points = generateDataFast(
            eq_infix,
            n_points=num_points,
            n_vars=num_vars,
            decimals=8,
            min_x=-10,
            max_x=10,
        )

        if points is None:
            return None

        structure = {
            "points": points.tolist(),
            "eq": eq_infix,
            "skeleton": eq_skeleton,
            "traversal": traversal,
            "eq_id": eq_id,
        }

        del (
            eq_no_consts,
            consts_elemns,
            w_const,
            wout_consts,
            eq_string,
            eq_infix,
            exps,
            eq_sympy_infix,
            eq_skeleton,
            traversal,
            points,
        )

        return structure

    except TimeoutError:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def monitor_memory():
    process = psutil.Process()  # Get the current process
    mem_info = process.memory_info()
    total_memory = mem_info.rss  # Resident Set Size (RSS) - memory used in bytes

    # Get memory usage of child processes (workers)
    for child in process.children(recursive=True):
        child_mem_info = child.memory_info()
        total_memory += child_mem_info.rss

    print(f"Total memory usage: {total_memory / (1024 ** 2):.2f} MB")  # Convert to MB


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    mode = args.mode
    if mode not in ["train", "val"]:
        print("Mode must be 'train' or 'val'")
        exit()

    # Load config
    cfg = load_cfg('cfg.yaml')
    num_points = cfg.num_points
    batch_size = cfg.Dataset.batch_size
    num_vars = cfg.num_vars
    num_skeletons = cfg.Dataset.num_train_skeletons if mode == "train" else cfg.Dataset.num_val_skeletons
    num_per_eq = cfg.Dataset.num_per_eq
    batch_ub = num_skeletons // batch_size

    if args.batch < 0 or args.batch >= batch_ub:
        print(f"Batch number must be between 0 and {batch_ub - 1}")
        exit()

    raw_data_folder = "../NeuralSymbolicRegressionThatScales/data/raw_datasets"
    dataset_path = f"{raw_data_folder}/{num_skeletons}"
    metadata = load_metadata_hdf5(hydra.utils.to_absolute_path(dataset_path))

    folder = f"{cfg.Dataset.dataset_folder}/{num_vars}_var/{mode}"
    os.makedirs(folder, exist_ok=True)

    n_eqs = metadata.total_number_of_eqs
    keys = metadata.word2id.keys()

    _results = []
    file_ID = 0

    seq2action = get_seq2action(cfg)

    for eq_id in tqdm(range(args.batch * batch_size, (args.batch + 1) * batch_size)):
        eq = load_eq(dataset_path, eq_id, metadata.eqs_per_hdf)
        results = Parallel(n_jobs=num_per_eq)(
            delayed(process_equation)(eq, keys, eq_id, cfg.Dataset.constants)
            for _ in range(num_per_eq)
        )
        _results.extend(results)

        del eq, results

        if (eq_id+1) % 100 == 0:

            points_list = []
            prefix_list = []
            action_list = []
            eq_list = []
            map_list = []
            map_idx = 0

            for structure in _results:

                if structure is None:
                    continue

                points = structure['points']
                points_list.append(np.expand_dims(points, axis=0))

                traversal = structure['traversal']

                for step_id in range(len(traversal)):

                    prefix_array = np.full((cfg.max_step,), np.nan)
                    for i in range(step_id):
                        prefix_array[i] = seq2action[traversal[i]]

                    action = seq2action[traversal[step_id]]

                    prefix_list.append(prefix_array)
                    action_list.append(action)
                    eq_list.append(structure['eq_id'])
                    map_list.append(map_idx)

                map_idx += 1

            with h5py.File(f'{folder}/{args.batch}_{file_ID}.h5', 'w') as hf:
                hf.create_dataset('points', data=np.vstack(points_list))
                hf.create_dataset('prefix', data=np.vstack(prefix_list))
                hf.create_dataset('eq_id', data=np.array(eq_list))
                hf.create_dataset('action', data=np.array(action_list))
                hf.create_dataset('map', data=np.array(map_list))

            _results.clear()

            file_ID += 1

    # monitor_memory()
