import os
import json
import argparse
import torch
import pytorch_lightning as pl
import wandb

from src.dataloader import Dataloader
from src.model import Model
from src.utils import check_git_status, create_experiment_dir, save_args, load_args_from_json
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Train STS model")
    parser.add_argument(
        "--config",
        default="../config.json",
        type=str,
        help="Path to JSON config file",
        required=True,
    )
    json_args = parser.parse_args()
    config = load_args_from_json(json_args.config)
    commit_id = check_git_status()
    experiment_dir = create_experiment_dir()

    model = Model(config["model_name"], config["learning_rate"])
    dataloader = Dataloader(
        config["model_name"],
        config["batch_size"],
        config["shuffle"],
        config["train_path"],
        config["dev_path"],
        config["test_path"],
        config["predict_path"],
        config["num_workers"],
    )
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"], default_root_dir=experiment_dir
    )

    save_args(config, experiment_dir)
    with open(os.path.join(experiment_dir, "git_commit.txt"), "w") as f:
        f.write(f"Git Commit ID: {commit_id}\n")
    wandb.init(project="sts-task", dir=experiment_dir)

    trainer.fit(model, dataloader)

    model_save_path = os.path.join(experiment_dir, "model.pt")
    torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    main()
