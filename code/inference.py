"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
import torch
import numpy as np
import urllib3

from typing import Callable, Dict, List, NoReturn, Tuple
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval, DenseRetrievalPolyEncoder
from trainer_qa import QuestionAnsweringTrainer
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
from utils_qa import check_no_error, postprocess_qa_predictions
from kiwipiepy import Kiwi 
from nori_tokenizer.nori import create_nori
from nori_tokenizer.elasticsearch_bm25 import create_es_connection, create_index, get_index_settings
from itertools import tee
from collections import defaultdict

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_train = True
    
    
    args = parser.parse_args()

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )

    # Kiwi 초기화 
    kiwi = Kiwi()
    
    # 불용어 목록 설정
    stopwords = {
        "이", "그", "저", "것", "수", "그리고", "그러나", "또한", "하지만", "즉", "또", "의", 
        "가", "을", "를", "에", "에게", "에서", "로", "부터", "까지", "와", "과", "도", "은", 
        "는", "이것", "그것", "저것", "뭐", "왜", "어떻게", "어디", "누구", "있다", "없다"
    }
    
    # tokenize_fn 정의
    def kiwipiepy_tokenize(text):        
        try:
            cleaned_text = text.encode('utf-8', 'ignore').decode('utf-8')
            tokens = kiwi.tokenize(cleaned_text)
            # 불용어 제거와 특정 품사 필터링 (예: 명사, 형용사만 사용)
            unigrams = [
                token.form for token in tokens 
                if token.form not in stopwords 
                and token.tag in {'NNG', 'NNP', 'VA'}
            ]
            # 명사만 선택
            nouns = [token[0] for token in tokens if token[1] == 'NNG' or token[1] == 'NNP']
            
            # 명사 + 명사 bigram 생성
            bigrams = [' '.join((nouns[i], nouns[i+1])) for i in range(len(nouns)-1)]            
            
            token_forms = unigrams + bigrams
            
            return token_forms
        except (UnicodeDecodeError, AttributeError) as e:
            print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
            # 오류 발생 시 빈 리스트를 반환하여 해당 지문을 무시
            return []
        except Exception as e:
            # 예상치 못한 다른 에러 발생 시 처리
            print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
            return [] 
        
    def nori_tokenize(text): 
        try: 
            return create_nori(text, es, INDEX)
        except (UnicodeDecodeError, AttributeError) as e:
            print(f"유니코드 디코딩 오류 발생, 지문 건너뛰기: {e}")
            # 오류 발생 시 빈 리스트를 반환하여 해당 지문을 무시
            return []
        except Exception as e:
            # 예상치 못한 다른 에러 발생 시 처리
            print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
            return [] 

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    
    # poly_encoder = DenseRetrievalPolyEncoder(
    #     model_name_or_path=args.model_name_or_path,
    #     data_path=args.data_path,
    #     context_path=args.context_path, 
    #     poly_m=64,  # You can adjust this value based on your needs
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )

    # True일 경우 : run passage retrieval
    # Elasticsearch 연결
    # 인덱스 생성
    # settings = get_index_settings()
    # INDEX = "bm25_tokenizer"
    
    # es = create_es_connection(INDEX)
    
    # create_index(es, INDEX, settings)

    if data_args.eval_retrieval: 
        datasets = run_retrieval(
            kiwipiepy_tokenize, kiwipiepy_tokenize, datasets, training_args, data_args,
        )
        
    # if training_args.do_train_poly:
        # train_polyencoder_with_bm25(poly_encoder) 

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    q_tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, q_tokenize_fn=q_tokenize_fn, data_path=data_path, context_path=context_path
    )

    # if data_args.use_faiss:
    #     retriever.build_faiss(num_clusters=data_args.num_clusters)
    #     df = retriever.retrieve_faiss(
    #         datasets["validation"], topk=data_args.top_k_retrieval
    #     )
    # else:
    
    df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval) 
    
    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
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

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # context의 일부가 아닌 offset_mapping을 None으로 설정하여 토큰 위치가 컨텍스트의 일부인지 여부를 쉽게 판별할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples

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

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
        top_k: int = 5  # top-k passages를 사용할 수 있도록 설정
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
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
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
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

# def train_polyencoder_with_bm25(poly_encoder):
#     poly_encoder.train() 
    
#     for batch in dataloader: 
#         optimizer.zero_grad() 
#         queries, positives, negatives = batch["query"], batch["positive"], batch["negative"] 
        
#         # Step 1: BM25 기반의 긍정 및 부정 문서 선택
#         pos_embeddings = poly_encoder.encode_docs(positives)
#         neg_embeddings = poly_encoder.encode_docs(negatives)
#         query_embeddings = poly_encoder.encode_query(queries)
        
#         # Step 2: Similarity 계산
#         pos_scores = (query_embeddings * pos_embeddings).sum(dim=-1)
#         neg_scores = (query_embeddings.unsqueeze(1) * neg_embeddings).sum(dim=-1) 
        
#         # Step 3: Loss 계산 및 학습 진행
#         loss = custom_loss_function(pos_scores, neg_scores)  # 사용자 정의 손실 함수 사용
#         loss.backward()
#         optimizer.step()

if __name__ == "__main__":
    main()
