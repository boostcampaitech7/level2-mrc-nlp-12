import json

# JSON 파일 불러오기
with open("../predictions_compare_10.json", "r", encoding="utf-8") as f:
    top_10_data = json.load(f)

with open("../predictions_compare_20.json", "r", encoding="utf-8") as f:
    top_20_data = json.load(f)

# ID를 기준으로 비교하여 조건을 만족하는 경우만 필터링
incorrect_in_top20 = []
for top10_item in top_10_data:
    top20_item = next(
        (item for item in top_20_data if item["id"] == top10_item["id"]), None
    )
    if (
        top20_item
        and top10_item["true_answer"] == top10_item["predicted_answer"]
        and top20_item["true_answer"] != top20_item["predicted_answer"]
    ):
        incorrect_in_top20.append(
            {
                "id": top10_item["id"],
                "context": top10_item["context"],
                "question": top10_item["question"],
                "true_answer": top10_item["true_answer"],
                "top_10_predicted": top10_item["predicted_answer"],
                "top_20_predicted": top20_item["predicted_answer"],
            }
        )

# 결과를 새로운 JSON 파일로 저장
with open("incorrect_in_top20.json", "w", encoding="utf-8") as f:
    json.dump(incorrect_in_top20, f, ensure_ascii=False, indent=4)

print(
    f"{len(incorrect_in_top20)}개의 항목이 top-10에서는 정답이지만, top-20에서는 오답으로 나타났습니다."
)

# top-10에서는 오답인데 top-20에서는 정답인 경우 찾기
correct_in_top20 = []
for top10_item in top_10_data:
    top20_item = next(
        (item for item in top_20_data if item["id"] == top10_item["id"]), None
    )
    if (
        top20_item
        and top10_item["true_answer"] != top10_item["predicted_answer"]
        and top20_item["true_answer"] == top20_item["predicted_answer"]
    ):
        correct_in_top20.append(
            {
                "id": top10_item["id"],
                "context": top10_item["context"],
                "question": top10_item["question"],
                "true_answer": top10_item["true_answer"],
                "top_10_predicted": top10_item["predicted_answer"],
                "top_20_predicted": top20_item["predicted_answer"],
            }
        )

# 결과를 새로운 JSON 파일로 저장
with open("correct_in_top20.json", "w", encoding="utf-8") as f:
    json.dump(correct_in_top20, f, ensure_ascii=False, indent=4)

print(
    f"{len(correct_in_top20)}개의 항목이 top-10에서는 오답이지만, top-20에서는 정답으로 나타났습니다."
)
