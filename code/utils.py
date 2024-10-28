import json
import os
import sys
from datetime import datetime, timedelta, timezone

import git
from arguments import DataTrainingArguments, ModelArguments
from transformers import HfArgumentParser, TrainingArguments


def check_git_status():
    repo = git.Repo(search_parent_directories=True)
    if repo.is_dirty():
        raise Exception(
            "Uncommitted changes in the repository. Commit or stash changes before running the experiment."
        )
    return repo.head.commit.hexsha


def create_experiment_dir(base_dir="../experiments", experiment_type=""):
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M%S_" + experiment_type)
    experiment_dir = os.path.join(base_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def save_args(args_dict, experiment_dir, commit_id):
    args_path = os.path.join(experiment_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)

    with open(os.path.join(experiment_dir, "git_commit.txt"), "w") as f:
        f.write(f"Git Commit ID: {commit_id}\n")


def load_args_from_json(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The JSON file '{json_file}' was not found.")
    with open(json_file, "r") as f:
        args_dict = json.load(f)
    return args_dict


def get_arguments(experiment_dir):
    # Initialize the parser
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    args_json_path = "../args.json"
    if os.path.exists(args_json_path):
        json_args = load_args_from_json(args_json_path)
    else:
        json_args = {}

    # Ensure output_dir is set to experiment_dir
    json_args["output_dir"] = experiment_dir

    # Parse command-line arguments
    parser.set_defaults(**json_args)
    combined_args = get_combined_args(json_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=combined_args
    )

    return model_args, data_args, training_args, json_args


def get_inference_arguments(experiment_dir):
    # Initialize the parser
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    args_json_path = "../args_inference.json"
    if os.path.exists(args_json_path):
        json_args = load_args_from_json(args_json_path)
    else:
        json_args = {}

    # Ensure output_dir is set to experiment_dir
    json_args["output_dir"] = experiment_dir
    json_args["data_path"] = json_args["model_name_or_path"]

    # Parse command-line arguments
    parser.set_defaults(**json_args)
    combined_args = get_combined_args(json_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=combined_args
    )

    return model_args, data_args, training_args, json_args


def get_combined_args(json_args):
    json_args_list = []
    for key, value in json_args.items():
        # Handle boolean arguments
        if isinstance(value, bool):
            if value:
                json_args_list.append(f"--{key}")
            else:
                # For boolean flags, the absence of the flag means False, so we can skip it
                pass
        else:
            json_args_list.append(f"--{key}")
            json_args_list.append(str(value))

    # Combine json_args_list with sys.argv[1:], giving precedence to command-line arguments
    # Command-line arguments come after to override json_args
    combined_args = json_args_list + sys.argv[1:]
    return combined_args
