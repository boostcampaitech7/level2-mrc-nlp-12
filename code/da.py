from datasets import load_dataset
import pandas as pd


from datasets import load_from_disk
import pandas as pd

t_dataset = load_from_disk("/data/ephemeral/home/level2-mrc-nlp-12/data/train_dataset")
train = t_dataset["train"]

wiki = pd.read_json("/data/ephemeral/home/level2-mrc-nlp-12/data/wikipedia_documents.json").transpose()
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

# 'answers' 필드를 train과 같은 형식으로 변환
def transform_answers(row):
    return {
        "text": row["answers"]["text"] if "text" in row["answers"] else [],
        "answer_start": row["answers"]["answer_start"] if "answer_start" in row["answers"] else []
    }

sample_df["answers"] = sample_df.apply(transform_answers, axis=1)

# train에 필요한 모든 열을 포함하도록 맞춤
columns_needed = ["id", "title", "context", "question", "answers"]
for col in columns_needed:
    if col not in sample_df.columns:
        if col == "title":
            sample_df[col] = ""  # title은 빈 문자열로 채움
        else:
            sample_df[col] = None

# 다시 데이터셋 형식으로 변환
sample_dataset_fixed = Dataset.from_pandas(sample_df)

# train 데이터와 sample_dataset_fixed 병합
combined_dataset = train.concatenate(sample_dataset_fixed)

# dataset에서 context 부분을 wiki에 추가하기 위해 context 부분을 pandas로 변환
context_df = sample_df[["context"]]

# wiki 데이터에 context 추가 후 중복 제거
wiki_combined = pd.concat([wiki, context_df], ignore_index=True)
wiki_unique = wiki_combined.drop_duplicates(subset="context")

# 중복 제거된 데이터를 새로운 json 파일로 저장
wiki_unique.to_json("./data/wikipedia_documents_combined_unique.json", orient='records', force_ascii=False)

# 새로운 train dataset 저장
combined_dataset.save_to_disk("/data/ephemeral/home/level2-mrc-nlp-12/data/train_dataset_combined")