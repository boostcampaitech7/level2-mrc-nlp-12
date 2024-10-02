import os
import git
import json
from datetime import datetime

def check_git_status():
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        raise Exception("Uncommitted changes in the repository. Commit or stash changes before running the experiment.")
    return repo.head.commit.hexsha

def create_experiment_dir(base_dir="../experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_args(args, experiment_dir):
    args_path = os.path.join(experiment_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(args, f, indent=4)

def load_args_from_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)
