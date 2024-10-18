# 51573
import os 
import json 


# 데이터를 파일에서 로드 
with open(os.path.join("../data", "wikipedia_documents.json"), "r", encoding="utf-8") as f: 
    wiki = json.load(f)

print("wiki: ", list(wiki.values())[51573])