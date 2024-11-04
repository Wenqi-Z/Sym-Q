import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


from SymQ import SymQ
from util import fix_seed
from agent import DQNAgent
from wrapper import load_dataset
from symbolic_world import SymbolicWorldEnv

import warnings

warnings.filterwarnings("ignore")


def main(args, cfg, point_set, skeleton):
    # Set up the environment
    env = SymbolicWorldEnv(cfg=cfg)

    # Set up the device for training
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Initialize and load the pretrained model
    pretrain_model = SymQ(cfg, device).to(device)

    if args.resume_path:
        state_dict = torch.load(args.resume_path, map_location=device)
        pretrain_model.load_state_dict(state_dict)

        # Separate parameters for different parts of the model
        model_params = list(pretrain_model.parameters())
        encoder_params = (
            list(pretrain_model.set_encoder.parameters())
            + list(pretrain_model.tree_encoder.parameters())
            + list(pretrain_model.linear1.parameters())
            + list(pretrain_model.linear2.parameters())
        )
        q_params = [
            p for p in model_params if not any(id(p) == id(e) for e in encoder_params)
        ]

        # Fix the encoder parameters (no training)
        for param in encoder_params:
            param.requires_grad = False

        # Initialize optimizer for training
        optimizer = optim.Adam(q_params, lr=args.lr)

    else:
        optimizer = optim.Adam(pretrain_model.parameters(), lr=args.lr)

    # Initialize the DDQN agent
    agent = DQNAgent(pretrain_model, optimizer, env.action_space.n)

    point_set = point_set.transpose(0, 1).unsqueeze(0)

    state, info = env.reset(point_set, skeleton, skeleton, None)
    done = False
    initial_reward = None
    reward_list = [0]
    log = {}

    # Initial guess
    while not done:
        action = agent.act(state, explore=False)
        next_state, reward, done, _, info = env.step(action.item())
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            initial_reward = reward.item()
            if reward > 0:
                reward_list.append(initial_reward)
                agent.memory.push_episode()
            else:
                agent.memory.discard_episode()

    # No need for few shot if the inital guess is good enough
    if initial_reward > 0.99:
        return (
            initial_reward,
            initial_reward,
            info["agent_expr"].expr,
            {},
            initial_reward,
        )

    # Start few shot
    for e in range(args.num_episodes):
        state, _ = env.reset(point_set, skeleton, skeleton, None)
        done = False
        steps = 0
        reward = 0

        while not done:
            if args.resume_path:
                action = agent.act(state, explore=(steps <= 10))
            else:
                action = agent.act(state, explore=True)

            next_state, reward, done, _, info = env.step(action.item())
            agent.remember(state, action, reward, next_state, done)

            steps += 1
            state = next_state

            if done:
                if (
                    reward.item() >= 0.6 * max(reward_list)
                    and reward.item() not in reward_list
                ):
                    reward_list.append(reward.item())
                    agent.memory.push_episode()
                else:
                    agent.memory.discard_episode()

        loss = agent.replay(args.batch_size, max(reward_list), 0.99)

        log[e] = {
            "Reward": reward.item(),
            "Expr": str(info["agent_expr"].expr),
            "Loss": loss,
        }

    if len(reward_list) == 1:
        return 0, 0, "none", log, max(reward_list)

    # Enforce the agent to converge
    last_loss = np.Inf
    for _ in range(100):
        loss = agent.replay(args.batch_size, max(reward_list))
        if abs(last_loss - loss) < 1e-4:
            break
        last_loss = loss

    # Final Expression
    state, info = env.reset(point_set, skeleton, skeleton, None)
    done = False
    while not done:
        action = agent.act(state, explore=False)
        next_state, reward, done, _, info = env.step(action.item())
        state = next_state

        if done:
            log["final"] = {
                "Reward": reward.item(),
                "Expr": str(info["agent_expr"].expr),
            }

    return initial_reward, reward.item(), info["agent_expr"].expr, log, max(reward_list)


if __name__ == "__main__":
    import os
    import yaml
    import json
    import argparse
    from tqdm import tqdm

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="SymQ+")
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--resume_path", type=str, default="", help="Path to resume the model from"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="Gpu id")
    parser.add_argument(
        "--is_dummy", type=int, default=0, help="Whether training from scratch"
    )
    args = parser.parse_args()

    folder_path = os.path.dirname(args.resume_path)

    # Load configuration from YAML file
    cfg = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)

    # Update config with command-line arguments
    cfg.update(vars(args))

    # Find unrecovered expressions
    r2_result = json.load(open(f"{folder_path}/beam_search_SSDNC_R2.json", "r"))
    selected_tests = []
    for eq_id, result in r2_result.items():
        if "R2" in r2_result[eq_id].keys() and r2_result[eq_id]["R2"] != 1:
            selected_tests.append(int(eq_id))

    SSDNC_dataset = load_dataset("SSDNC", cfg)
    few_shot_result = {}

    for eq_id in tqdm(selected_tests):
        # Set seeds for reproducibility
        fix_seed(args.seed)
        point_set, _, _, _, _, skeleton, _, expr = SSDNC_dataset[eq_id]
        initial_reward, fewshot_reward, agent_expr, log, explore_max = main(
            args, cfg, point_set, skeleton
        )
        few_shot_result[eq_id] = {}
        few_shot_result[eq_id]["expr"] = str(expr)
        few_shot_result[eq_id]["initial_reward"] = initial_reward
        few_shot_result[eq_id]["fewshot_reward"] = fewshot_reward
        few_shot_result[eq_id]["agent_expr"] = str(agent_expr)
        few_shot_result[eq_id]["max"] = explore_max
        few_shot_result[eq_id]["log"] = log

        if args.is_dummy:
            json.dump(
                few_shot_result,
                open(f"{folder_path}/few_shot_result_dummy.json", "w"),
                indent=4,
            )
        else:
            json.dump(
                few_shot_result,
                open(f"{folder_path}/few_shot_result.json", "w"),
                indent=4,
            )
