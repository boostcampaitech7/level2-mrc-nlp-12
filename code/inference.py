"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple

import numpy as np
import torch
import urllib3
from arguments import DataTrainingArguments, ModelArguments
from custom_logger import CustomLogger
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from elasticsearch import Elasticsearch
from retrieval import SparseRetrieval
from trainer_qa import QuestionAnsweringTrainer, prepare_validation_features
from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    set_seed,
)
from utils import (
    check_git_status,
    create_experiment_dir,
    get_arguments,
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
    experiment_dir = create_experiment_dir(commit_id=commit_id)
    model_args, data_args, training_args, json_args = get_arguments(experiment_dir)

    logger.set_config()
    logger.set_training_args(training_args=training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(seed=training_args.seed, deterministic=training_args.deterministic)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    datasets = load_from_disk(data_args.dataset_name)

    tokenizer, model = get_tokenizer_and_model(model_args)

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize,
            datasets,
            training_args,
            data_args,
            model_args.data_path,
            model_args.context_path,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

    # Save the final arguments
    save_args(json_args, experiment_dir, commit_id)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    q_tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
    es: Elasticsearch = None,
    INDEX: str = "",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn,
        data_path=data_args.data_path,
        context_path=data_args.context_path,
        q_tokenize_fn=q_tokenize_fn,
        es=es,
        INDEX=INDEX,
    )

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve_es(
            datasets["validation"], topk=data_args.top_k_retrieval, es=es
        )

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset,
            test_examples=datasets["validation"],
            data_args=data_args,
            column_names=column_names,
            datasets=datasets,
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate(
            data_args=data_args, column_names=column_names, datasets=datasets
        )
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


def custom_loss_function(pos_scores, neg_scores, margin=1.0):
    # Positive similarity를 최대화하고, Negative similarity를 최소화하도록 손실 계산
    positive_loss = 1 - pos_scores
    negative_loss = torch.clamp(neg_scores - margin, min=0.0)

    # 두 손실의 평균을 구함
    loss = positive_loss.mean() + negative_loss.mean()
    return loss


if __name__ == "__main__":
    main()
