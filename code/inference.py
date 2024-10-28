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
from retrieval import SparseRetrieval
from trainer_qa import QuestionAnsweringTrainer
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
    prepare_validation_features,
)

try:
    from elasticsearch import Elasticsearch
    from kiwipiepy import Kiwi
    from nori_tokenizer.elasticsearch_bm25 import (
        create_es_connection,
        create_index,
        get_index_settings,
    )
    from nori_tokenizer.nori import create_nori
    from retrieval import SparseRetrieval_ElasticSearch
    from utils import okt_tokenize

    kiwi = Kiwi()

    def kiwipiepy_tokenize(text):
        try:
            cleaned_text = text.encode("utf-8", "ignore").decode("utf-8")
            tokens = kiwi.tokenize(cleaned_text)
            unigrams = [
                token.form
                for token in tokens
                if token.form not in stopwords and token.tag in {"NNG", "NNP", "VA"}
            ]
            nouns = [
                token[0] for token in tokens if token[1] == "NNG" or token[1] == "NNP"
            ]

            bigrams = [
                " ".join((nouns[i], nouns[i + 1])) for i in range(len(nouns) - 1)
            ]
            token_forms = unigrams + bigrams
            return token_forms
        except (UnicodeDecodeError, AttributeError) as e:
            print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
            return []
        except Exception as e:
            print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
            return []

    from konlpy.tag import Okt

    okt = Okt()

    def okt_tokenize(text):
        try:
            cleaned_text = text.encode("utf-8", "ignore").decode("utf-8")
            tokens = okt.pos(cleaned_text)

            # 불용어 제거와 특정 품사 필터링 (예: 명사, 형용사만 사용)
            meaningful_token_forms = [
                token[0]
                for token in tokens
                if token[0] not in stopwords
                and token[1]
                in {
                    "Noun",
                    "Adjective",
                    "Verb",
                    "Adverb",
                    "ProperNoun",
                    "Determiner",
                }
            ]

            token_forms = [token[0] for token in tokens]
            return meaningful_token_forms + token_forms
        except (UnicodeDecodeError, AttributeError) as e:
            print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
            return []
        except Exception as e:
            print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
            return []

    from nori_tokenizer.elasticsearch_bm25 import (
        create_es_connection,
        create_index,
        get_index_settings,
    )
    from nori_tokenizer.nori import create_nori

    settings = get_index_settings()
    index_name = "bm25_tokenizer"
    es = create_es_connection(index_name)
    create_index(es, index_name, settings)

    def nori_tokenize(text, es, index_es):
        try:
            return create_nori(text, es, index_es)
        except (UnicodeDecodeError, AttributeError) as e:
            print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
            # 오류 발생 시 빈 리스트를 반환하여 해당 지문을 무시
            return []
        except Exception as e:
            # 예상치 못한 다른 에러 발생 시 처리
            print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
            return []

    stopwords = {
        "이",
        "그",
        "저",
        "것",
        "수",
        "그리고",
        "그러나",
        "또한",
        "하지만",
        "즉",
        "또",
        "의",
        "가",
        "을",
        "를",
        "에",
        "에게",
        "에서",
        "로",
        "부터",
        "까지",
        "와",
        "과",
        "도",
        "은",
        "는",
        "이것",
        "그것",
        "저것",
        "뭐",
        "왜",
        "어떻게",
        "어디",
        "누구",
        "있다",
        "없다",
    }
    elasticsearch_installed = True
except ImportError:
    elasticsearch_installed = False

logger = CustomLogger(name=__name__)


def main():
    commit_id = check_git_status()
    experiment_dir = create_experiment_dir(experiment_type="inference")
    model_args, data_args, training_args, json_args = get_arguments(
        experiment_dir, experiment_type="inference"
    )

    logger.set_config()
    logger.set_training_args(training_args=training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(seed=training_args.seed)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    datasets = load_from_disk(data_args.dataset_name)

    tokenizer, model = get_tokenizer_and_model(model_args)

    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        if elasticsearch_installed:
            settings = get_index_settings()
            index_es = "bm25_tokenizer"
            es = create_es_connection(index_es)
            create_index(es, index_es, settings)
            datasets = run_sparse_retrieval(
                okt_tokenize,
                okt_tokenize,
                datasets,
                training_args,
                data_args,
                es,
                index_es,
            )

        else:
            datasets = run_sparse_retrieval(
                tokenize_fn=tokenizer.tokenize,
                datasets=datasets,
                training_args=training_args,
                data_args=data_args,
                data_path=model_args.data_path,
                context_path=model_args.context_path,
            )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

    # Save the final arguments
    save_args(json_args, experiment_dir, commit_id)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    data_path: str = "../models",
    context_path: str = "../data/wikipedia_documents.json",
    datasets: DatasetDict = None,
    training_args: TrainingArguments = None,
    data_args: DataTrainingArguments = None,
    q_tokenize_fn: Callable[[str], List[str]] = None,
    es=None,
    index: str = "",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    if elasticsearch_installed:
        retriever = SparseRetrieval_ElasticSearch(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
            q_tokenize_fn=q_tokenize_fn,
            es=es,
            index=index,
        )
    else:
        retriever = SparseRetrieval(
            tokenize_fn=tokenize_fn,
            data_path=data_path,
            context_path=context_path,
        )
        retriever.get_sparse_embedding()
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    extra_columns = set(df.columns) - {"id", "question", "context", "answers"}
    if extra_columns:
        print(f"Removing extra columns: {extra_columns}")
        df = df.drop(columns=list(extra_columns))

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
            }
        )
        df = df[["id", "question", "context"]]
    elif training_args.do_eval:
        f = Features(
            {
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
            }
        )
        if "answers" not in df.columns:
            df["answers"] = None
        df = df[["id", "question", "context", "answers"]]
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
        fn_kwargs={
            "tokenizer": tokenizer,
            "column_names": column_names,
            "max_seq_length": max_seq_length,
            "data_args": data_args,
        },
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
