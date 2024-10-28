import random

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

t_dataset = load_from_disk("/data/ephemeral/home/level2-mrc-nlp-12/data/train_dataset")
train = t_dataset["train"]

wiki = pd.read_json(
    "/data/ephemeral/home/level2-mrc-nlp-12/data/wikipedia_documents.json"
).transpose()
dataset = load_dataset("squad_kor_v1")

# corpus = list(set([example["context"] for example in dataset["train"]]))
# print(f"총 {len(corpus)}개의 지문이 있습니다.")
# wiki_set = list(set(wiki["text"]))
# print(f"갯수: {len(wiki_set)}")
# overlap = set(wiki_set) & set(corpus)
# print(f"wiki_set의 지문 중 corpus에 있는 지문의 수: {len(overlap)}개")

# 8000개의 랜덤 샘플 추출
random.seed(42)  # 재현성을 위해 seed 설정
sample_dataset = dataset["train"].shuffle(seed=42).select(range(8000))

# sample_dataset을 pandas DataFrame으로 변환
sample_df = sample_dataset.to_pandas()

c_sampled_df = []
# 'answers' 필드를 train과 같은 형식으로 변환
for idx in range(len(sample_df)):
    new_example = {
        "title": sample_df.iloc[idx]["title"],
        "context": sample_df.iloc[idx]["context"],
        "question": sample_df.iloc[idx]["question"],
        "id": sample_df.iloc[idx]["id"],
        "answers": {
            "text": sample_df.iloc[idx]["answers"]["text"],
            "answer_start": sample_df.iloc[idx]["answers"]["answer_start"].astype(
                np.int64
            ),
        },
        "document_id": None,
    }
    c_sampled_df.append(new_example)
print("train 추가 데이터셋")
print(c_sampled_df[0:5])
# 다시 데이터셋 형식으로 변환
c_sampled_df = pd.DataFrame(c_sampled_df)
sample_dataset_fixed = Dataset.from_pandas(c_sampled_df)

# train 데이터와 sample_dataset_fixed 병합
combined_dataset = concatenate_datasets([train, sample_dataset_fixed])

print("train 병합")
print(combined_dataset[3950:3955])
# sample_dataset을 pandas DataFrame으로 변환하고 context 부분만 추출
sample_df = sample_dataset.to_pandas()

# wiki 데이터 포맷에 맞게 context 데이터를 추가할 빈 데이터프레임 생성
new_rows = pd.DataFrame(
    {
        "text": sample_df["context"],  # context 내용을 text로 추가
        "corpus_source": None,  # 나머지 값들은 비워둠 (None 또는 기본값)
        "url": "TODO",
        "domain": None,
        "title": None,
        "author": None,
        "html": None,
        "document_id": None,
    }
)

# 기존의 wiki 데이터에 새로운 행 추가
combined_df = pd.concat([wiki, new_rows], ignore_index=True)

# 중복된 'text' 제거
wiki_unique = combined_df.drop_duplicates(subset="text")
# wiki_unique = wiki_unique.transpose()

# 중복 제거된 데이터를 새로운 json 파일로 저장
wiki_unique.to_json(
    "../data/wikipedia_documents_combined.json", orient="index", force_ascii=False
)

# 새로운 train dataset 저장
combined_dataset.save_to_disk(
    "/data/ephemeral/home/level2-mrc-nlp-12/data/train_dataset_combined"
)
