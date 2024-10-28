import json
import os
import random
import sys
from typing import NoReturn

import numpy as np
import torch
import wandb
from arguments import DataTrainingArguments, ModelArguments
from custom_logger import CustomLogger
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import (
    QuestionAnsweringTrainer,
    prepare_train_features,
    prepare_validation_features,
)
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils import (
    check_git_status,
    create_experiment_dir,
    get_arguments,
    get_data_collator,
    get_tokenizer_and_model,
    print_examples_on_evaluation,
    save_args,
)
from utils_qa import (
    check_no_error,
    post_processing_function,
    postprocess_qa_predictions,
)

logger = CustomLogger(name=__name__)


def main():
    commit_id = check_git_status()
    experiment_dir = create_experiment_dir(experiment_type="train")

    model_args, data_args, training_args, json_args = get_arguments(experiment_dir)
    logger.set_config()
    logger.set_training_args(training_args=training_args)

    set_seed(seed=training_args.seed, deterministic=training_args.deterministic)

    print(model_args.model_name_or_path)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # wandb 설정
    wandb.init(
        project=training_args.WANDB_PROJECT,
        name=training_args.run_name if training_args.run_name else None,
    )

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    tokenizer, model = get_tokenizer_and_model(model_args)

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(
            data_args, training_args, model_args, datasets, tokenizer, model, logger
        )

    # Save the final arguments
    save_args(json_args, experiment_dir, commit_id)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    logger,
) -> NoReturn:
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    # do_train과 do_eval을 동시에 실행한 경우?
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        column_names = datasets["train"].column_names
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "tokenizer": tokenizer,
                "column_names": column_names,
                "max_seq_length": max_seq_length,
                "training_args": training_args,
                "data_args": data_args,
            },
        )
    elif training_args.do_eval:
        eval_dataset = datasets["validation"]
        column_names = datasets["validation"].column_names

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            fn_kwargs={
                "tokenizer": tokenizer,
                "column_names": column_names,
                "max_seq_length": max_seq_length,
                "training_args": training_args,
                "data_args": data_args,
            },
        )

    # Data collator
    data_collator = get_data_collator(tokenizer=tokenizer, training_args=training_args)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )
    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            data_args=data_args, column_names=column_names, datasets=datasets
        )

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # validation 데이터에 대해 predict
        predictions = trainer.predict(
            test_dataset=eval_dataset,
            test_examples=datasets["validation"],
            data_args=data_args,
            column_names=column_names,
            datasets=datasets,
        )
        preds = predictions.predictions
        labels = predictions.label_ids

        # dataset, prediction, answer, score을 묶어서 results에 할당
        results = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            pred_text = pred["prediction_text"]
            true_text = label["answers"]["text"][0]

            score = metric.compute(predictions=[pred], references=[label])

            results.append(
                {
                    "question": datasets["validation"][i]["question"],
                    "context": datasets["validation"][i]["context"],
                    "prediction": pred_text,
                    "answer": true_text,
                    "f1": score["f1"],
                    "em": score["exact_match"],
                }
            )
        print_examples_on_evaluation(results)


if __name__ == "__main__":
    main()
    wandb.finish()  # finish the wandb run
