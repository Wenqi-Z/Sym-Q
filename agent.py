from collections import namedtuple, deque
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

# Define a transition tuple for storing experiences
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DequeDataset(Dataset):
    def __init__(self, data_deque):
        self.data_deque = data_deque

    def __len__(self):
        return len(self.data_deque)

    def __getitem__(self, idx):
        return (
            self.data_deque[idx].state["point_set"].squeeze(),
            self.data_deque[idx].state["tree"].squeeze(),
            self.data_deque[idx].action,
            self.data_deque[idx].reward,
        )


class EpisodeMemory(object):
    """A simple memory buffer for storing and sampling experiences."""

    def __init__(self, capacity=30):
        """Initialize the memory buffer with a fixed capacity."""
        self.memory = deque([], maxlen=capacity)
        self.reward = None

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the memory."""
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def clear(self):
        """Clear the memory."""
        self.reward = None
        self.memory.clear()

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)


class ReplayMemory(object):
    """A simple memory buffer for storing and sampling experiences."""

    def __init__(self, capacity=10000000):
        """Initialize the memory buffer with a fixed capacity."""
        self.memory = deque([], maxlen=capacity)
        self.e_memory = EpisodeMemory()

    def push(self, state, action, reward, next_state, done):
        self.e_memory.push(state, action, reward, next_state, done)

    def push_episode(self):
        """Store a transition in the memory."""
        reward = self.e_memory.memory[-1].reward
        for state, action, _, next_state, done in reversed(self.e_memory.memory):
            self._push(state, action, reward, next_state, done)
        self.e_memory.clear()

    def discard_episode(self):
        self.e_memory.clear()

    def _push(self, state, action, reward, next_state, done):
        """Store a transition in the memory."""
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)


class DQNAgent:
    def __init__(self, model, optimizer, action_size):
        """Initialize the agent with a model, optimizer, and action size."""
        self.action_size = action_size
        self.memory = ReplayMemory()  # Experience replay buffer
        self.model = model

        self.device = model.device
        self.optimizer = optimizer

        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def remember(self, state, action, reward, next_state, done):
        """Store an experience in replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, explore=False):
        """Return an action based on the current state and policy."""

        point_set = state["point_set"].to(self.device)
        tree = state["tree"].to(self.device)

        self.model.eval()
        action, q_values = self.model.act(point_set, tree)

        if not explore:
            return action
        else:
            act_prob = F.softmax(q_values, dim=1)
            act_dist = Categorical(act_prob)
            action = act_dist.sample()
            return action

    def replay(self, batch_size, max_r, th=0.6):
        """Train the model using a batch of experiences from the memory."""
        if len(self.memory) == 0:
            return None  # Not enough samples for training

        dataset = DequeDataset(self.memory.memory)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0
        n_update = 0
        self.model.eval()

        for point_set_batch, tree_batch, actions, rewards in dataloader:
            n_update += 1
            point_set_batch = point_set_batch.to(self.device)
            tree_batch = tree_batch.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            actions = actions

            q_k, _, _, _ = self.model(point_set_batch, tree_batch, point_set_batch)

            loss_ce = self.ce_loss(q_k.squeeze(), actions.squeeze())
            ratio_reward = torch.clamp(rewards - th * max_r, min=0)
            loss = (ratio_reward.squeeze() * loss_ce.squeeze()).mean()

            # Updata the model
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / n_update
