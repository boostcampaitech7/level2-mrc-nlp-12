from elasticsearch import Elasticsearch

# Elasticsearch에 연결
es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=("elastic", "S+p+aDSIG=YduplYogRt"),
    verify_certs=False  # SSL 인증서 무시
)

# Shingle 필터가 포함된 인덱스 생성
index_name = "my_shingle_index"

# 인덱스 생성
settings = {
  "settings": {
    "analysis": {
      "tokenizer": {
        "nori_tokenizer": {
          "type": "nori_tokenizer",
          "decompound_mode": "mixed"  # 복합어를 형태소 단위로 분리하는 옵션
        }
      },
      "filter": {
        "my_shingle_filter": {
          "type": "shingle",
          "min_shingle_size": 1,
          "max_shingle_size": 3,
          "output_unigrams": True
        }
      },
      "analyzer": {
        "korean_shingle_analyzer": {
          "type": "custom",
          "tokenizer": "nori_tokenizer",
          "filter": [
            "lowercase",
            "my_shingle_filter"
          ]
        }
      }
    }
  },
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "korean_shingle_analyzer"
      }
    }
  }
}

# 인덱스 생성 요청
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=settings)
    print(f"Index {index_name} created successfully")
else:
    print(f"Index {index_name} already exists")

# 문서 추가
doc = {
    "content": "I love programming and data science"
}

es.index(index=index_name, body=doc)

# 분석 요청 (Shingle 적용된 분석기 사용)
analyze_body = {
    "analyzer": "my_shingle_analyzer",
    "text": "어디 한번 한국어도 잘 되나 보자"
}

response = es.indices.analyze(index=index_name, body=analyze_body)

# 분석 결과 출력
print("Shingle tokens:")
for token in response['tokens']:
    print(token['token'])