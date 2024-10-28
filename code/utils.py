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
    # if repo.is_dirty():
    #     raise Exception(
    #         "Uncommitted changes in the repository. Commit or stash changes before running the experiment."
    #     )
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

    if model_args.use_kiwi:
        from kiwipiepy import Kiwi

        kiwi = Kiwi()

        def kiwipiepy_tokenize(text):
            try:
                cleaned_text = text.encode("utf-8", "ignore").decode("utf-8")
                tokens = kiwi.tokenize(cleaned_text)
                # 불용어 제거와 특정 품사 필터링 (예: 명사, 형용사만 사용)
                unigrams = [
                    token.form
                    for token in tokens
                    if token.form not in stopwords and token.tag in {"NNG", "NNP", "VA"}
                ]
                # 명사만 선택
                nouns = [
                    token[0]
                    for token in tokens
                    if token[1] == "NNG" or token[1] == "NNP"
                ]

                # 명사 + 명사 bigram 생성
                bigrams = [
                    " ".join((nouns[i], nouns[i + 1])) for i in range(len(nouns) - 1)
                ]

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

    if model_args.use_okt:
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
                # 오류 발생 시 빈 리스트를 반환하여 해당 지문을 무시
                return []
            except Exception as e:
                # 예상치 못한 다른 에러 발생 시 처리
                print(f"알 수 없는 오류 발생, 지문 건너뛰기: {e}")
                return []

    elif model_args.use_nori:
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

        def nori_tokenize(text, es, INDEX):
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

    if model_args.use_stopwords:
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

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    return tokenizer, model


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
