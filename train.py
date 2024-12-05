import os
import time
import shutil
import argparse
from itertools import chain
from datetime import datetime

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import wandb

from SymQ import SymQ
from utils import fix_seed
from loss import SupConLoss
from wrapper import load_dataset
from evaluation import evaluate_action, heat_map_evaluation


def main(args, cfg, weights_dir):
    # Training data
    train_dataset = load_dataset("train", cfg, debug=cfg["DEBUG"])
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Test data
    SSDNC_dataset = load_dataset("eval", cfg)

    # Initialize model and optimizer
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = SymQ(cfg, device).to(device)
    print(f"SymQ running on {device}")

    # Load resume path
    if args.resume_path:
        model.load_state_dict(torch.load(args.resume_path, map_location=device))

    # Loss
    CL = SupConLoss(device=device).to(device)
    CE = nn.CrossEntropyLoss()

    # Optimizer
    model_params = list(model.parameters())
    encoder_params = list(model.set_encoder.parameters()) + list(
        model.tree_encoder.parameters()
    )
    q_params = [p for p in model_params if all(id(p) != id(e) for e in encoder_params)]
    params_to_clip = chain(
        model.set_encoder.parameters(), model.tree_encoder.parameters()
    )
    optimizer = optim.Adam(
        [{"params": encoder_params, "lr": args.lr}, {"params": q_params, "lr": args.lr}]
    )

    # Evaluation initialization
    step_acc = 0
    ma_eq_acc = 0
    acc_top_1 = 0
    acc_top_2 = 0
    max_rank = 0

    # Initialize wandb
    if not cfg["DEBUG"]:
        wandb.init(project="SymQ", config=cfg)

    for epoch in range(args.num_epochs):

        for i, data in enumerate(dataloader):
            model.train()
            start_time = time.time()

            # Move data to device
            point_set, tree, action, support_points, labels = [
                d.to(device) for d in data[:5]
            ]

            # Forward pass
            q_values, point_embedding, support_embedding, _ = model(
                point_set, tree, support_points
            )

            # Concatenate embeddings
            CL_embedding = torch.stack(
                (
                    F.normalize(point_embedding, p=2, dim=1),
                    F.normalize(support_embedding, p=2, dim=1),
                ),
                dim=1,
            )

            # Compute loss
            points_contrastive_loss = CL(CL_embedding, labels)
            _, q_labels = action.max(dim=1)
            q_loss = CE(q_values, q_labels)
            loss = 0.5 * points_contrastive_loss + q_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(params_to_clip, max_norm=1.0)
            optimizer.step()

            print(
                f"Epoch {epoch + 1} update {i+1}, q_loss: {q_loss.item():.4f}, contrastive_loss:{points_contrastive_loss.item():.4f}, time:{time.time() - start_time:.4f} s"
            )

            # Evaluation
            if (i + 1) % 10 == 0:
                model.eval()
                step_acc = evaluate_action(model, train_dataset)

                (
                    max_rank,
                    acc_top_1,
                    acc_top_2,
                ) = heat_map_evaluation(model, SSDNC_dataset, cfg)

                ma_eq_acc = 0.9 * ma_eq_acc + 0.1 * acc_top_1

                if acc_top_1 >= ma_eq_acc:
                    model_save_name = f"{weights_dir}/best.pth"
                    torch.save(model.state_dict(), model_save_name)

                if not cfg["DEBUG"]:
                    wandb.log(
                        {
                            "Train/bce_loss": q_loss.item(),
                            "Train/points_contrastive_loss": points_contrastive_loss.item(),
                            "Evaluation/equation accuracy Top 1": acc_top_1,
                            "Evaluation/equation accuracy Top 2": acc_top_2,
                            "Evaluation/Max rank": max_rank,
                            "Evaluation/step accuracy": step_acc,
                        }
                    )

        if epoch % 5 == 0:
            model_save_name = f"{weights_dir}/epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_save_name)

    if not cfg["DEBUG"]:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SymQ")
    parser.add_argument(
        "--num_epochs", type=int, default=100_000, help="number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=float, default=2, help="seed")
    parser.add_argument("--resume_path", type=str, default="", help="resume_path")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
    parser.add_argument("--doc_string", type=str, default="")
    args = parser.parse_args()

    # Create a folder to save the weights
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    weights_dir = f"model/{start_time}"
    os.makedirs(weights_dir, exist_ok=True)
    print(f"This training will be saved in {weights_dir}")

    # Copy related files to the folder
    for file in [
        "cfg.yaml",
        "encoder.py",
        "evaluation.py",
        "SymQ.py",
        "train.py",
        "wrapper.py",
    ]:
        shutil.copyfile(file, f"{weights_dir}/{file}")

    # Load config file
    cfg = yaml.load(open("cfg.yaml", "r"), Loader=yaml.FullLoader)

    # Add args to cfg
    cfg["training"] = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "resume_path": args.resume_path,
        "doc": args.doc_string,
    }

    # Make the training reproducible
    fix_seed(args.seed)

    # Run training
    main(args, cfg, weights_dir)
