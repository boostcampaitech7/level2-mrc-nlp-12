# Open-Domain Question Answering

## 소개

"서울의 GDP는 세계 몇 위야?", "MRC가 뭐야?"

우리는 궁금한 것들이 생겼을 때, 아주 당연하게 검색엔진을 활용하여 검색을 합니다. 이런 검색엔진은 최근 기계독해 (MRC, Machine Reading Comprehension) 기술을 활용하며 매일 발전하고 있는데요. 본 대회에서는 우리가 당연하게 활용하던 검색엔진, 그것과 유사한 형태의 시스템을 만들어 볼 것입니다.

Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다.
다양한 QA 시스템 중, Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가되기 때문에 더 어려운 문제입니다.


본 ODQA 대회에서 우리가 만들 모델은 two-stage로 구성되어 있습니다. 첫 단계는 질문에 관련된 문서를 찾아주는 "retriever" 단계이고, 다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계입니다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 여러분들 손으로 직접 만들어보게 됩니다.

따라서, 대회는 더 정확한 답변을 내주는 모델을 만드는 팀이 좋은 성적을 거두게 됩니다.

최종적으로 테스트해야하는 결과물은
- input: Query만 주어집니다.
- output: 주어진 Query에 알맞는 string 형태의 답안

## 평가방법

두 가지 평가지표를 사용합니다.

1. Exact Match (EM): 모델의 예측과, 실제 답이 정확하게 일치할 때만 점수가 주어집니다. 즉 모든 질문은 0점 아니면 1점으로 처리됩니다. 단, 띄어쓰기나 "."과 같은 문자가 포함되어 있다고 오답으로 처리되면 억울하겠죠? 이런 것은 제외한 후 정답에 대해서만 일치하는지 확인합니다. 또한 답이 하나가 아닐 수 있는데, 이런 경우는 하나라도 일치하면 정답으로 간주합니다.



2. F1 Score: EM과 다르게 부분 점수를 제공합니다. 예를 들어, 정답은 "Barack Obama"지만 예측이 "Obama"일 때, EM의 경우 0점을 받겠지만 F1 Score는 겹치는 단어도 있는 것을 고려해 부분 점수를 받을 수 있습니다.



EM 기준으로 리더보드 등수가 반영되고, F1은 참고용으로만 활용됩니다.


## 세부일정

- 프로젝트 전체 기간 (4주) : 9월 30일 (월) 10:00 ~ 10월 24일 (목) 19:00
  - 팀 병합 기간 : 10월 1일 (화) 16:00 까지
- 리더보드 제출 오픈 : 10월 2일 (수) 10:00
- 리더보드 제출 마감 및 최종 점수 공개 (Private) : 10월 24일 (목) 19:00
- GPU 서버 운영 기간 : 9월 30일 (월) 10:00 ~ 10월 25일 (금) 16:00

## 룰

### 대회 룰

- [대회 참여 제한] NLP 도메인을 수강하고 있는 캠퍼에 한하여 리더보드 제출이 가능합니다.
- [팀 결성 기간] 팀 결성은 대회 페이지 공개 후 2일차 오후 4시까지 필수로 진행해 주세요. 팀이 완전히 결성되기 전까지는 리더보드 제출이 불가합니다.
- [일일 제출횟수] 일일 제출횟수는 '팀 단위 10회'로 제한합니다. (일일횟수 초기화 자정 진행)
- [외부 데이터셋 규정] KLUE-MRC 데이터셋을 제외한 모든 외부 데이터 사용 허용합니다.
- [기학습 가중치 사용] 기학습 가중치는 제한 없이 모두 허용하나, KLUE MRC 데이터로 학습된 기학습 가중치 (pretrained weight) 사용은 금지합니다. 가중치는 모두 public 에 공개되어 있고 저작권 문제 없이 누구나 사용 가능해야 합니다. 사용하는 기학습 가중치는 공지 게시판의 ‘기학습 가중치 사용 공지’ 게시글에 댓글로 가중치 및 접근 가능한 링크를 반드시 공유합니다. 이미 공유되어 있을 경우 추가로 공유주실 필요는 없습니다.
- [평가 데이터 활용] 학습 효율 측면에서 테스트셋을 분석하고 사용(학습)하는 행위는 본 대회에서는 금지합니다. (눈으로 직접 판별 후 라벨링 하는 행위 포함)
- [데이터셋 저작권] 대회 데이터셋은 '캠프 교육용 라이선스' 아래 사용 가능합니다. 저작권 관련 세부 내용은 부스트코스 공지사항을 반드시 참고 해주세요.

### AI Stages 대회 공통사항

- [Private Sharing 금지] 비공개적으로 다른 팀과 코드 혹은 데이터를 공유하는 것은 허용하지 않습니다.
코드 공유는 반드시 대회 게시판을 통해 공개적으로 진행되어야 합니다.
- [최종 결과 검증 절차] 리더보드 상위권 대상으로 추후 코드 검수가 필요한 대상으로 판단될 경우 개별 연락을 통해 추가 검수 절차를 안내드릴 수 있습니다. 반드시 결과가 재현될 수 있도록 최종 코드를 정리 부탁드립니다. 부정행위가 의심될 경우에는 결과 재현을 요구할 수 있으며, 재현이 어려울 경우 리더보드 순위표에서 제외될 수 있습니다.
- [공유 문화] 공개적으로 토론 게시판을 통해 모델링에 대한 아이디어 혹은 작성한 코드를 공유하실 것을 권장 드립니다. 공유 문화를 통해서 더욱 뛰어난 모델을 대회 참가자 분들과 같이 개발해 보시길 바랍니다.
- [대회 참가 기본 매너] 좋은 대회 문화 정착을 위해 아래 명시된 행위는 지양합니다.
  - 대회 종료를 앞두고 (3일 전) 높은 점수를 얻을 수 있는 전체 코드를 공유하는 행위
  - 타 참가자와 토론이 아닌 단순 솔루션을 캐내는 행위


## 파일 구성

### 저장소 구조

```
level2-mrc-nlp-12
├─ .gitignore
├─ .pre-commit-config.yaml
├─ README.md
├─ assets
│  ├─ dataset.png
│  └─ utils_qa.py
├─ data
│  ├─ test_dataset
│  │  ├─ dataset_dict.json
│  │  └─ validation
│  │     ├─ dataset_info.json
│  │     ├─ dataset.arrow
│  │     └─ state.json
│  ├─ train_dataset
│  │  ├─ dataset_dict.json
│  │  ├─ train
│  │  │  ├─ dataset_info.json
│  │  │  ├─ dataset.arrow
│  │  │  └─ state.json
│  │  └─ validation
│  │     ├─ dataset_info.json
│  │     ├─ dataset.arrow
│  │     └─ state.json
├─ code
│  ├─ arguments.py
│  ├─ custom_logger.py
│  ├─ ensemble.py
│  ├─ inference.py
│  ├─ requirements.txt
│  ├─ retrieval.py
│  ├─ train.py
│  ├─ trainer_qa.py
│  ├─ utils.py
│  └─ utils_qa.py
├─ pyproject.toml
└─ scripts
   ├─ eda.ipynb
   ├─ pororo.ipynb
   ├─ question_generation.py
   ├─ wikipedia_preprocessing_ADEA.ipynb
   └─ wikipedia_unique.py
```

## 설치 방법

### 요구 사항

```bash
git clone <repository>
cd <repository>
pip install -r code/requirements.txt
```


## 데이터 소개

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```bash
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다.

# 훈련, 평가, 추론

## train

실행결과는 `experiments/` 아래에 실행일시 및 파일명을 기준으로 디렉토리를 생성해 저장하고 있습니다.

이 때 arguments는 `args.json` 파일을 통해 입력할 수 있으며 실행 커맨드에서 직접 입력할 수도 있습니다. 같은 argument를 설정할 경우, cli 상의 입력이 우선시됩니다. 

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요.


```bash
# 학습 예시 (args.json 사용)
python train.py
```
```json
// args.json
{
  "do_train": true
}
```
또는 아래와 같이 실행할 수 있습니다.
```bash
# 학습 예시 (train_dataset 사용)
python train.py --output_dir ./models/train_dataset --do_train
``` 

## eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# 학습 예시 (args.json 사용)
python train.py
```
```json
// args.json
{
  "do_train": true,
  "do_eval": true
}
```

## inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다. train의 `args.json`과 마찬가지로 argument를 별도의 파일 `args_inference.py`로 입력할 수 있습니다.

* 학습한 모델의 test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다.

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

* 둘을 동시에 진행할 경우 eval이 정상적으로 마쳐지지 않는 오류가 발생하니 아직은 동시에 진행하지 않아야 합니다.

* 학습에 사용할 mrc 모델의 경로를 `model_name_or_path`로 지정해야 하며, retrieval 모델의 경로는 `data_path`로 지정해야 합니다. `data_path`에 retrieval 모델의 bin 파일이 없을 경우에는 해당 경로에 retrieval 파일을 저장합니다. 따라서 `model_name_or_path`와 `data_path`를 통일하는 것을 권장합니다.

* `context_path`는 retrieval에 사용할 passages 파일의 파일명을 포함한 경로를 입력합니다.

* `dataset_name`은 `--do_predict`인 경우 `../data/test_dataset`을, `--do_eval`인 경우 `../data/train_dataset`을 사용하는 것을 기본으로 합니다. `--do_eval`에 `dataset_name`을 명시하지 않을 경우 학습 데이터로 예측 결과를 만들기 때문에 리더보드에 사용할 수 없는 출력파일을 얻게 됩니다.

```bash
# ODQA 실행 (test_dataset 사용)
python inference.py 
```
```json
// args_inference.json
{
  "model_name_or_path": "../experiments/YYYYMMDD_hhmmss_train",
  "data_path": "../experiments/YYYYMMDD_hhmmss_train",
  "context_path": "../data/wikipedia_documents.json",
  "dataset_name": "../data/test_dataset",
  "do_predict": true
}
```

### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

## Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 기존 파일이 load 됩니다.

2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다.

3. `./outputs/` 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.


## pre-commit hooks

commit 할 때마다 `.pre-commit-config.yaml`에 정의한 행동을 실행해 코드의 스타일을 통일합니다. [black](https://github.com/psf/black), [isort](https://github.com/pycqa/isort) 및 [pre-defined hooks by Github](https://github.com/pre-commit/pre-commit-hooks)의 몇 가지 hooks를 사용하고 있습니다.

### 요구사항

- `pre-commit` 설치

```bash
$ pre-commit --version
pre-commit 4.0.1
$ pre-commit install

```

### Troubleshoot
- pre-commit이 실행되지 않을 경우
```bash
$ pre-commit clean && pre-commit install && pre-commit run --all-files
```