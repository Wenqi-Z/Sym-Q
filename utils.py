import threading

import numpy as np
import torch


seq_to_action = {
    "x_1": 0,
    "x_2": 1,
    "c": 2,
    "abs": 3,
    "add": 4,
    "mul": 5,
    "div": 6,
    "sqrt": 7,
    "exp": 8,
    "log": 9,
    "pow": 10,
    "sin": 11,
    "cos": 12,
    "tan": 13,
    "asin": 14,
    "acos": 15,
    "atan": 16,
    "sinh": 17,
    "cosh": 18,
    "tanh": 19,
    "coth": 20,
    "-3": 21,
    "-2": 22,
    "-1": 23,
    "0": 24,
    "1": 25,
    "2": 26,
    "3": 27,
    "4": 28,
    "5": 29,
}

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


def seq_to_tree(sequence, max_step):
    """
    Convert a sequence of operations into a tree representation.

    This function takes a sequence of operations and converts it into a tree-like
    structure of specified maximum steps, with each operation represented as a
    one-hot encoded vector.

    Args:
    sequence (list of str/int): A list of operations. Each operation can be
                                represented as a string or an integer.
    max_step (int): The maximum number of steps in the tree. This defines the
                    height of the tree.

    Returns:
    torch.Tensor: A tensor representing the tree. The tensor has a shape of
                  (max_step, len(seq_to_action)) and is of type float32. Each
                  row in the tensor represents a step in the tree, with the
                  corresponding operation encoded as a one-hot vector.
    """
    tree = torch.zeros((max_step, len(seq_to_action)), dtype=torch.float32)
    for i, op_seq in enumerate(sequence):
        tree[i, seq_to_action[op_seq]] = 1

    return tree


def handle_timeout(signum, frame):
    """
    A function to handle a timeout signal.

    This function is designed to be used as a handler for signal-based timeouts.
    When a specified signal is caught, this function raises a TimeoutError.
    Typically used in conjunction with signal.signal to handle, for example,
    the SIGALRM signal.

    Args:
    signum (int): The signal number.
    frame: Current stack frame (ignored in this function).

    Raises:
    TimeoutError: An error indicating that a timeout has occurred.

    Example:
    >>> import signal
    >>> signal.signal(signal.SIGALRM, handle_timeout)
    >>> signal.alarm(5)  # Set a 5-second alarm
    >>> # Your code here
    >>> signal.alarm(0)  # Disable the alarm

    Note:
    - This function is intended to be used as a signal handler and should not
      be called directly in normal circumstances.
    """
    raise TimeoutError("Time out")
