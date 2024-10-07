import numpy as np
import torch

import yaml
from box import Box


def load_cfg(path: str) -> Box:
    with open(path, "r") as file:
        # Load the YAML content
        cfg = Box(yaml.safe_load(file))
    return cfg


# seq_to_action = {
#     "x_1": 0,
#     "x_2": 1,
#     "c": 2,
#     "abs": 3,
#     "add": 4,
#     "mul": 5,
#     "div": 6,
#     "sqrt": 7,
#     "exp": 8,
#     "log": 9,
#     "pow": 10,
#     "sin": 11,
#     "cos": 12,
#     "tan": 13,
#     "asin": 14,
#     "acos": 15,
#     "atan": 16,
#     "sinh": 17,
#     "cosh": 18,
#     "tanh": 19,
#     "coth": 20,
#     "-3": 21,
#     "-2": 22,
#     "-1": 23,
#     "0": 24,
#     "1": 25,
#     "2": 26,
#     "3": 27,
#     "4": 28,
#     "5": 29,
# }


BENCHMARK = {
    "Nguyen_1": "x_1**3 + x_1**2 + x_1",
    "Nguyen_2": "x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_3": "x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_4": "x_1**6 + x_1**5 + x_1**4 + x_1**3 + x_1**2 + x_1",
    "Nguyen_5": "sin(x_1**2)*cos(x_1) - 1",
    "Nguyen_6": "sin(x_1) + sin(x_1 + x_1**2)",
    "Nguyen_7": "log(x_1+1)+log(x_1**2+1)",
    "Nguyen_8": "x_1**0.5",
    "Nguyen_9": "sin(x_1) + sin(x_2**2)",
    "Nguyen_10": "2*sin(x_1)*cos(x_2)",
    "Nguyen_11": "x_1**x_2",
    "Nguyen_12": "x_1**4-x_1**3+0.5*x_2**2-x_2",
    "Keijzer_3": "0.3 * x_1 * sin(2*pi*x_1)",
    "Keijzer_4": "pow(x_1,3)*exp(-x_1)*cos(x_1)*sin(x_1)*(pow(sin(x_1),2)*cos(x_1)-1)",
    "Keijzer_9": "log(x_1+sqrt(pow(x_1,2)+1))",
    "Keijzer_11": "x_1*x_2+sin((x_1-1)*(x_2-1))",
    "Keijzer_13": "6*sin(x_1)*cos(x_2)",
    "Keijzer_14": "8/(2+x_1**2+x_2**2)",
    "Keijzer_15": "0.2*x_1**3+0.5*x_2**3-x_2-x_1",
    "Constant_1": "3.39*pow(x_1,3)+2.12*pow(x_1,2)+1.78*x_1",
    "Constant_2": "sin(pow(x_1,2))*cos(x_1)-0.75",
    "Constant_3": "sin(1.5*x_1)*cos(0.5*x_2)",
    "Constant_4": "2.7*pow(x_1,x_2)",
    "Constant_5": "sqrt(1.23*x_1)",
    "Constant_6": "pow(x_1,0.426)",
    "Constant_7": "2*sin(1.3*x_1)*cos(x_2)",
    "Constant_8": "log(x_1+1.4)+log(pow(x_1,2)+1.3)",
    "R_1": "(x_1+1)**3/(x_1**2-x_1+1)",
    "R_2": "(x_1**5-3*x_1**3+1)/(x_1**2+1)",
    "R_3": "(x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1)",
    "Jin_1": "2.5*x_1**4-1.3*x_1**3+0.5*x_2**2-1.7*x_2",
    "Feynman_1": "exp(-x_1**2/2)/sqrt(2*pi)",
    "Feynman_2": "exp(-(x_1/x_2)**2/2)/(sqrt(2*pi)*x_2)",
    "Feynman_3": "x_1*x_2",
    "Feynman_4": "x_1*x_2",
    "Feynman_5": "0.5*x_1*x_2**2",
    "Feynman_6": "x_1/x_2",
    "Feynman_7": "1.5*x_1*x_2",
    "Feynman_8": "x_1/(4*3.14*x_2**2)",
    "Feynman_9": "(x_1*x_2**2)/2",
    "Feynman_10": "x_1*x_2**2",
    "Feynman_11": "x_1/(2*(1+x_2))",
    "Feynman_12": "x_1*x_2/(2*3.14)",
}


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seq_to_tree(sequence, cfg):
    max_step = cfg.max_step
    seq2action = get_seq2action(cfg)
    tree = torch.zeros((max_step, len(seq2action)), dtype=torch.float32)
    for i, op_seq in enumerate(sequence):
        tree[i, op_seq] = 1

    return tree


def handle_timeout(signum, frame):
    raise TimeoutError("Time out")


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def comp_wn(model):
    K = 0
    N = 0
    for p in model.parameters():
        N += torch.sum(p**2)
        K += p.numel()
    return torch.sqrt(N / K)


def get_seq2action(cfg):
    num_vars = cfg.num_vars
    if num_vars < 1 or num_vars > 3:
        raise ValueError("Invalid number of variables")

    seq2action = {
        "c": 0,
        "abs": 1,
        "add": 2,
        "mul": 3,
        "div": 4,
        "sqrt": 5,
        "exp": 6,
        "ln": 7,
        "pow": 8,
        "sin": 9,
        "cos": 10,
        "tan": 11,
        "-3": 12,
        "-2": 13,
        "-1": 14,
        "0": 15,
        "1": 16,
        "2": 17,
        "3": 18,
        "4": 19,
        "5": 20,
        "asin": 21,
        "acos": 22,
        "atan": 23,
        "sinh": 24,
        "cosh": 25,
        "tanh": 26,
        "coth": 27,
        "x_1": 28,
    }

    if num_vars == 2:
        seq2action["x_2"] = 29
    elif num_vars == 3:
        seq2action["x_2"] = 29
        seq2action["x_3"] = 30

    return seq2action
