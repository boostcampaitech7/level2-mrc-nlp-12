import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import git
from arguments import DataTrainingArguments, ModelArguments
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
)


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


def get_data_collator(tokenizer, training_args):
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    return DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )


def get_tokenizer_and_model(model_args):
    """
    AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    """
    config = AutoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name is not None
            else model_args.model_name_or_path
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        (
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path
        ),
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    return config, tokenizer, model


def print_examples_on_evaluation(results):
    # 정답을 맞춘것과 틀린것을 10개씩 출력 (f1 score 기준)
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    print("*** 상위 10개 예측 ***")
    for result in results_sorted[:10]:
        print(f"Context: {result['context'][:50]}...")
        print(f"Question: {result['question']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Answer: {result['answer']}")
        print(f"f1: {result['f1']}")
        print(f"em: {result['em']}")
        print()

    print("*** 하위 10개 예측 ***")
    for result in results_sorted[-10:]:
        print(f"Context: {result['context'][:50]}...")
        print(f"Question: {result['question']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Answer: {result['answer']}")
        print(f"f1: {result['f1']}")
        print(f"em: {result['em']}")
        print()

    # 같은 context에 대한 질문들 출력
    print("*** 같은 context에 대한 질문 pair ***")
    context_to_results = defaultdict(list)
    for result in results_sorted:
        context = result["context"]
        context_to_results[context].append(result)

    for context, results in context_to_results.items():
        if len(results) > 1:
            print(f"Context: {context[:50]}...")
            for result in results:
                print(f"  Question: {result['question']}")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Answer: {result['answer']}")
                print(f"  f1: {result['f1']}")
                print(f"  em: {result['em']}")
                print()
