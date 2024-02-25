from abc import ABC, abstractmethod
import random
import warnings

import torch
import numpy as np
import sympy as sp
from sympy import lambdify, sympify

from bfgs import bfgs
import signal

# def handle_timeout(signum, frame):
#     raise TimeoutError

# signal.signal(signal.SIGALRM, handle_timeout)

# Module-level constants
PLACEHOLDER = "PH"
OPT_SEQ = {
    "x_1": "x_1",
    "x_2": "x_2",
    "c": "c",
    "Abs": "abs",
    "+": "add",
    "*": "mul",
    "/": "div",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "**": "pow",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "coth": "coth",
    "-3": "-3",
    "-2": "-2",
    "-1": "-1",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
}


class Node:
    UNARY_OPERATORS = {
        "sin",
        "cos",
        "tan",
        "exp",
        "sqrt",
        "log",
        "Abs",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "coth",
    }
    BINARY_OPERATORS = {"+", "*", "/", "**"}

    def __init__(self, parent=None):
        self.val = PLACEHOLDER
        self.parent = parent

    def __str__(self):
        return self.val

    @property
    def binary(self) -> bool:
        """Check if the node is a binary operation."""
        if self.val in Node.UNARY_OPERATORS:
            return False
        elif self.val in Node.BINARY_OPERATORS:
            return True
        return None

    def set_value(self, val: str) -> list:
        """
        Set the value of the node and expand the placeholders accordingly.

        Returns:
        - A list of the newly created placeholder nodes.
        """
        assert isinstance(val, str), "Node value must be a string."

        self.val = val

        if self.binary is False:
            self.child = Node(self)
            return [self.child]
        elif self.binary is True:
            self.left = Node(self)
            self.right = Node(self)
            return [self.right, self.left]  # LIFO: right first

        return []


class Expression(ABC):
    pass


class TargetExpression(Expression):
    def __init__(
        self,
        point_set: torch.FloatTensor,
        expr: str = None,
        skeleton: str = None,
        opt_sequence: list = None,
    ):
        """
        Initialize the Expression with a default value or an existing expression.
        """
        assert point_set.shape[2] == 3, "Point set must be 3D."
        self._point_set = point_set  # 1 x N x 3
        self._expr = expr
        self._skeleton = skeleton
        self._opt_sequence = opt_sequence

    @property
    def point_set(self) -> torch.FloatTensor:
        return self._point_set

    @property
    def expr(self) -> str:
        return self._expr

    @property
    def skeleton(self) -> str:
        return self._skeleton

    @property
    def opt_sequence(self) -> list:
        return self._opt_sequence


class AgentExpression(Expression):
    def __init__(self, point_set: torch.FloatTensor):
        """
        Initialize the Expression with a default value or an existing expression.
        """
        # Initialize the expression
        self.tree = Node()
        self._expr = None
        # reference point set
        assert point_set.shape[2] == 3, "Point set must be 3D."
        self._ref_point_set = point_set  # 1 x N x 3
        self._point_set = None
        # record the history of agent operations
        self._opt_sequence = []
        # used to get the nodes to work on for current operation
        self._to_be_expanded = [self.tree]
        self.expr = None

    @property
    def skeleton(self) -> str:
        return self._update_skeleton(self.tree)

    @property
    def opt_sequence(self) -> list:
        return self._opt_sequence

    def add_opt(self, opt: str) -> int:
        """
        Add operation to the expression.

        Returns:
        - A flag indicating the status of the expression:
            - 0: operation successfully added.
            - 1: no more placeholder.
        """

        if not self._to_be_expanded:
            raise ValueError("No placeholder found in the expression.")

        node = self._to_be_expanded.pop()

        # Set the value of the node and add the newly created placeholders
        h_list = node.set_value(opt)
        self._to_be_expanded.extend(h_list)

        # Add the operation to the history
        self._opt_sequence.append(OPT_SEQ[opt])

        return 0 if self._to_be_expanded else 1

    def _update_skeleton(self, node: Node):
        """Update the string representation of the expression."""
        if node.binary is False:
            return f"{node.val}({self._update_skeleton(node.child)})"
        if node.binary is True:
            return f"({self._update_skeleton(node.left)} {node.val} {self._update_skeleton(node.right)})"
        return node.val


if __name__ == "__main__":
    pass
