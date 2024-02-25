import json

from tqdm import tqdm
import numpy as np
import sympy as sp
import torch
import torch.nn.functional as F


def evaluate_SSDNC(model, env, dataset, save_file=None):
    # Evaluation of the performance on SSDNC
    device = model.device
    num_tests = len(dataset)
    record = {}

    for i in tqdm(range(num_tests)):
        point_set, _, _, _, _, skeleton, seq, expr = dataset[i]
        point_set = point_set.to(device).unsqueeze(0).transpose(1, 2)
        obs, info = env.reset(point_set, expr, skeleton, seq)
        done = False

        while not done:
            point_set = obs["point_set"]
            tree = obs["tree"]
            action, _ = model.act(point_set, tree)

            obs, _, done, _, info = env.step(action.item())

        target_expr = info["target_expr"]
        agent_expr = info["agent_expr"]

        record[str(i)] = {
            "target": target_expr.skeleton,
            "agent_expr": agent_expr.skeleton,
            "skeleton_correct": sp.sympify(target_expr.skeleton)
            == sp.sympify(agent_expr.skeleton),
            "R2_match": info["R2"] > 0.99 if info["R2"] is not None else False,
        }

        if save_file:
            with open(save_file, "w") as file:
                json.dump(record, file, indent=4)

    return record


def evaluate_action(model, dataset, num_tests=100):
    # Evaluation on steps
    device = model.device
    point_set_list = []
    tree_list = []
    action_list = []

    for _ in range(num_tests):
        idx = np.random.randint(0, len(dataset))
        point_set, tree, action, _, _, _, _, _ = dataset[idx]
        point_set_list.append(point_set)
        tree_list.append(tree)
        action_list.append(action)

    point_set = torch.stack(point_set_list).to(device)
    tree = torch.stack(tree_list).to(device)
    action = torch.stack(action_list).to(device)

    # Compute action
    gt_action = action.argmax(dim=1)
    agent_action, _ = model.act(point_set, tree)

    # Compute accuracy
    accuray = (gt_action == agent_action).float().mean(dim=0).item()

    return accuray


def evaluate_operation(model, dataset, cfg):
    device = model.device
    num_actions = cfg["SymQ"]["num_actions"]
    operation_correct = np.zeros(num_actions)
    operation_counts = np.zeros(num_actions)
    choice_distribution = np.zeros((num_actions, num_actions))
    num_tests = len(dataset)
    step_analysis = {}

    for idx in tqdm(range(num_tests)):

        step_error = []  # 1: error / 0: correct

        for data in dataset[idx]:
            point_set, tree, action, _, _, _, _, _ = data

            point_set = point_set.to(device)
            tree = tree.to(device)
            action = action.to(device)

            gt_action = action.argmax(dim=0)
            agent_action, _ = model.act(point_set, tree)

            # Update counts
            operation_counts[gt_action] += 1

            if gt_action == agent_action:
                operation_correct[gt_action] += 1
                step_error.append(0)
            else:
                step_error.append(1)

            choice_distribution[gt_action, agent_action] += 1

        step_length = len(step_error)
        if step_length not in step_analysis.keys():
            step_analysis[step_length] = {
                "total": 1,
                "error": np.array(step_error),
            }
        else:
            step_analysis[step_length]["total"] += 1
            step_analysis[step_length]["error"] += np.array(step_error)

    # Compute final accuracy for each operation
    accuracy = operation_correct / operation_counts

    accuracy = np.nan_to_num(accuracy)

    for step_length in step_analysis:
        step_analysis[step_length]["error"] = step_analysis[step_length][
            "error"
        ].tolist()

    return (
        operation_correct,
        operation_counts,
        accuracy,
        choice_distribution,
        step_analysis,
    )


def n_tests(model, dataset, cfg):
    n_tests = len(dataset)
    max_step = cfg["max_step"]

    # Initialization of different heat maps
    heat_map = -1 * torch.ones((max_step, n_tests))
    thred_heat_map = -1 * torch.ones((max_step, n_tests))
    rank_heat_map = -1 * torch.ones((max_step, n_tests))
    rank_heat_map_2 = 10 * torch.ones((max_step, n_tests))
    difference_heat_map = -1 * torch.ones((max_step, n_tests))
    max_rank = 0

    for idx in tqdm(range(n_tests)):
        data = dataset[idx]

        point_set_list = []
        tree_list = []
        action_list = []

        for d in data:
            point_set, tree, action, _, _, _, _, _ = d

            point_set_list.append(point_set)
            tree_list.append(tree)
            action_list.append(action)

        point_set = torch.cat(point_set_list, dim=0).to(model.device)
        tree = torch.cat(tree_list, dim=0).to(model.device)
        action = torch.stack(action_list).to(model.device)

        # Predict actions and compute softmax values
        gt_action = action.argmax(dim=1)
        agent_action, value = model.act(point_set, tree)
        row_softmax = F.softmax(value, dim=1)

        # Calculate differences and ranks for softmax values
        max_vals_per_row, _ = torch.max(row_softmax, dim=1, keepdim=True)
        differences = max_vals_per_row - row_softmax

        selected_softmax_values_differences = differences.gather(
            1, gt_action.unsqueeze(1)
        ).squeeze()
        selected_softmax_values = row_softmax.gather(1, gt_action.unsqueeze(1))

        ranks = row_softmax.argsort(dim=1, descending=True).argsort(
            dim=1, descending=False
        )
        selected_ranks = ranks.gather(1, gt_action.unsqueeze(1)).squeeze()
        selected_ranks_2 = selected_ranks.clone()

        # Update max rank if necessary
        if selected_ranks.dim() > 0:
            max_val = torch.max(selected_ranks).item()
        else:
            max_val = selected_ranks.item()
        if max_val > max_rank:
            max_rank = max_val

        # Determine the mask for top-K accuracy and update heat maps
        k = 1
        mask = selected_ranks <= k  # Top-K acc
        selected_ranks[mask] = 1
        selected_ranks[~mask] = 0
        selected_softmax_values = selected_softmax_values.squeeze()
        mask = selected_softmax_values_differences < 0.1
        selected_softmax_values_differences[mask] = 1
        selected_softmax_values_differences[~mask] = 0

        # Compute accuracy
        result = (gt_action == agent_action).float()
        heat_map[: (len(result)), idx] = result
        init_3_fixed_heat_map = heat_map.clone()
        difference_heat_map[: (len(result)), idx] = selected_softmax_values_differences
        thred_heat_map[: (len(result)), idx] = selected_softmax_values
        rank_heat_map[: (len(result)), idx] = selected_ranks
        rank_heat_map_2[: (len(result)), idx] = selected_ranks_2
        init_3_fixed_heat_map[:3] = rank_heat_map[:3]

    # Post-processing to calculate accuracies and percentages
    mask = heat_map != -1
    all_ones = (heat_map == 1) | (heat_map == -1)
    valid_columns = all_ones.all(dim=0) & mask.any(dim=0)
    count = valid_columns.sum().item()
    acc_once = count / n_tests
    mask = rank_heat_map != -1
    all_ones = (rank_heat_map == 1) | (rank_heat_map == -1)
    valid_columns = all_ones.all(dim=0) & mask.any(dim=0)
    count = valid_columns.sum().item()
    acc_top_2 = count / n_tests

    return max_rank, acc_once, acc_top_2


def heat_map_evaluation(model, dataset, cfg):
    n_tests = len(dataset)
    max_step = cfg["max_step"]

    # Initialization of different heat maps
    heat_map = -1 * torch.ones((max_step, n_tests))
    thred_heat_map = -1 * torch.ones((max_step, n_tests))
    rank_heat_map = -1 * torch.ones((max_step, n_tests))
    rank_heat_map_2 = 10 * torch.ones((max_step, n_tests))
    difference_heat_map = -1 * torch.ones((max_step, n_tests))
    max_rank = 0

    for idx in tqdm(range(n_tests)):
        data = dataset[idx]

        point_set_list = []
        tree_list = []
        action_list = []

        for d in data:
            point_set, tree, action, _, _, _, _, _ = d

            point_set_list.append(point_set)
            tree_list.append(tree)
            action_list.append(action)

        point_set = torch.cat(point_set_list, dim=0).to(model.device)
        tree = torch.cat(tree_list, dim=0).to(model.device)
        action = torch.stack(action_list).to(model.device)

        # Predict actions and compute softmax values
        gt_action = action.argmax(dim=1)
        agent_action, value = model.act(point_set, tree)
        row_softmax = F.softmax(value, dim=1)

        # Calculate differences and ranks for softmax values
        max_vals_per_row, _ = torch.max(row_softmax, dim=1, keepdim=True)
        differences = max_vals_per_row - row_softmax

        selected_softmax_values_differences = differences.gather(
            1, gt_action.unsqueeze(1)
        ).squeeze()
        selected_softmax_values = row_softmax.gather(1, gt_action.unsqueeze(1))

        ranks = row_softmax.argsort(dim=1, descending=True).argsort(
            dim=1, descending=False
        )
        selected_ranks = ranks.gather(1, gt_action.unsqueeze(1)).squeeze()
        selected_ranks_2 = selected_ranks.clone()

        # Update max rank if necessary
        if selected_ranks.dim() > 0:
            max_val = torch.max(selected_ranks).item()
        else:
            max_val = selected_ranks.item()
        if max_val > max_rank:
            max_rank = max_val

        # Determine the mask for top-K accuracy and update heat maps
        k = 1
        mask = selected_ranks <= k  # Top-K acc
        selected_ranks[mask] = 1
        selected_ranks[~mask] = 0
        selected_softmax_values = selected_softmax_values.squeeze()
        mask = selected_softmax_values_differences < 0.1
        selected_softmax_values_differences[mask] = 1
        selected_softmax_values_differences[~mask] = 0

        # Compute accuracy
        result = (gt_action == agent_action).float()
        heat_map[: (len(result)), idx] = result
        init_3_fixed_heat_map = heat_map.clone()
        difference_heat_map[: (len(result)), idx] = selected_softmax_values_differences
        thred_heat_map[: (len(result)), idx] = selected_softmax_values
        rank_heat_map[: (len(result)), idx] = selected_ranks
        rank_heat_map_2[: (len(result)), idx] = selected_ranks_2
        init_3_fixed_heat_map[:3] = rank_heat_map[:3]

    # Post-processing to calculate accuracies and percentages
    mask = heat_map != -1
    all_ones = (heat_map == 1) | (heat_map == -1)
    valid_columns = all_ones.all(dim=0) & mask.any(dim=0)
    count = valid_columns.sum().item()
    acc_once = count / 963
    mask = rank_heat_map != -1
    all_ones = (rank_heat_map == 1) | (rank_heat_map == -1)
    valid_columns = all_ones.all(dim=0) & mask.any(dim=0)
    count = valid_columns.sum().item()
    acc_top_2 = count / 963
    is_one = heat_map == 1
    count_ones = is_one.sum()
    percentage_ones = 0

    return max_rank, acc_once, acc_top_2, percentage_ones


def heat_map(model, dataset, cfg):
    n_tests = len(dataset)

    heat_map = -1 * torch.ones((cfg["max_step"], n_tests))
    thred_heat_map = -1 * torch.ones((cfg["max_step"], n_tests))
    rank_heat_map = -1 * torch.ones((cfg["max_step"], n_tests))
    rank_heat_map_2 = 10 * torch.ones((cfg["max_step"], n_tests))
    difference_heat_map = -1 * torch.ones((cfg["max_step"], n_tests))
    max_rank = 0

    for idx in tqdm(range(n_tests)):
        data = dataset[idx]

        point_set_list = []
        tree_list = []
        action_list = []

        for d in data:
            point_set, tree, action, _, _, _, _, _ = d

            point_set_list.append(point_set)
            tree_list.append(tree)
            action_list.append(action)

        point_set = torch.cat(point_set_list, dim=0).to(model.device)
        tree = torch.cat(tree_list, dim=0).to(model.device)
        action = torch.stack(action_list).to(model.device)

        # Compute action idx
        gt_action = action.argmax(dim=1)
        agent_action, value = model.act(point_set, tree)
        row_softmax = F.softmax(value, dim=1)

        max_vals_per_row, _ = torch.max(row_softmax, dim=1, keepdim=True)
        differences = max_vals_per_row - row_softmax

        selected_softmax_values_differences = differences.gather(
            1, gt_action.unsqueeze(1)
        ).squeeze()
        selected_softmax_values = row_softmax.gather(1, gt_action.unsqueeze(1))

        ranks = row_softmax.argsort(dim=1, descending=True).argsort(
            dim=1, descending=False
        )
        selected_ranks = ranks.gather(1, gt_action.unsqueeze(1)).squeeze()
        selected_ranks_2 = selected_ranks.clone()

        if selected_ranks.dim() > 0:
            max_val = torch.max(selected_ranks).item()
        else:
            max_val = selected_ranks.item()
        if max_val > max_rank:
            max_rank = max_val

        k = 1
        mask = selected_ranks <= k  # Top-K acc
        selected_ranks[mask] = 1
        selected_ranks[~mask] = 0
        selected_softmax_values = selected_softmax_values.squeeze()

        # Compute accuracy
        result = (gt_action == agent_action).float()
        heat_map[: (len(result)), idx] = result
        init_3_fixed_heat_map = heat_map.clone()
        difference_heat_map[: (len(result)), idx] = selected_softmax_values_differences
        thred_heat_map[: (len(result)), idx] = selected_softmax_values
        rank_heat_map[: (len(result)), idx] = selected_ranks
        rank_heat_map_2[: (len(result)), idx] = selected_ranks_2
        init_3_fixed_heat_map[:3] = rank_heat_map[:3]

    is_one = heat_map == 1
    not_minus_one = heat_map != -1
    count_ones = is_one.sum(dim=1)
    count_not_minus_one = not_minus_one.sum(dim=1)
    percentage_ones = count_ones.float() / count_not_minus_one.float()
    percentage_ones[torch.isnan(percentage_ones)] = 1
    count_ones = is_one.sum()
    count_not_minus_one = not_minus_one.sum()
    percentage_ones = count_ones.float() / count_not_minus_one.float()

    return (
        heat_map.detach().cpu().numpy(),
        difference_heat_map.detach().cpu().numpy(),
        rank_heat_map.detach().cpu().numpy(),
        rank_heat_map_2.detach().cpu().numpy(),
        init_3_fixed_heat_map.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    import argparse
    import os
    import yaml

    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from SymQ import SymQ
    from wrapper import load_dataset
    from symbolic_world import SymbolicWorldEnv

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--metric", type=str, default="", help="evaluation metric")
    args = parser.parse_args()

    folder_path = os.path.dirname(args.weights_path)

    # Load config file
    cfg = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)

    # Initialize model and optimizer
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = SymQ(cfg, device).to(device)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    print(f"SymQ running on {device}")

    # Evaluate SSDNC recovery rate
    if args.metric == "recovery":
        print("Evaluating SSDNC recovery rate...")
        save_file = f"{folder_path}/SSDNC_recovery.json"
        env = SymbolicWorldEnv(cfg)
        dataset = load_dataset("SSDNC", cfg)
        record = evaluate_SSDNC(model, env, dataset, save_file)
        with open(save_file, "w") as file:
            json.dump(record, file, indent=4)

    # Evaluate operation accuracy
    if args.metric == "operation":
        print("Evaluating operation accuracy...")
        from utils import seq_to_action

        action_to_seq = {v: k for k, v in seq_to_action.items()}
        dataset = load_dataset("eval", cfg)
        operation_correct, operation_counts, accuracy, choice_dist, step_analysis = (
            evaluate_operation(model, dataset, cfg)
        )
        opt_acc = {
            "num_operation": {
                action_to_seq[i]: operation_counts[i] for i in range(len(action_to_seq))
            },
            "num_correct": {
                action_to_seq[i]: operation_correct[i]
                for i in range(len(action_to_seq))
            },
            "accuracy": {
                action_to_seq[i]: accuracy[i] for i in range(len(action_to_seq))
            },
            "choice_distribution": {
                action_to_seq[i]: {
                    action_to_seq[j]: choice_dist[i, j]
                    for j in range(len(action_to_seq))
                }
                for i in range(len(action_to_seq))
            },
            "step_analysis": step_analysis,
        }
        with open(f"{folder_path}/operation_error.json", "w") as file:
            json.dump(opt_acc, file, indent=4)

    if args.metric == "heatmap":
        print("Evaluating SSDNC heat map...")
        dataset = load_dataset("eval", cfg)
        m1, m2, m3, m4, m5 = heat_map(model, dataset, cfg)
        heatmaps = {
            "Heat Map": m1.tolist(),
            "Difference Heat Map": m2.tolist(),
            "MCTS Heat Map": m3.tolist(),
            "Action Rank Heat Map": m4.tolist(),
            "Fixed Heat Map": m5.tolist(),
        }
        with open(f"{folder_path}/heatmaps.json", "w") as file:
            json.dump(heatmaps, file, indent=4)
