# elasticsearch_bm25.py
import json
from elasticsearch import Elasticsearch

def create_index(es, index_name, settings):
    """
    Elasticsearch 인덱스를 생성하는 함수
    """
    if not es.indices.exists(index=index_name):
        return es.indices.create(index=index_name, body=settings)
    else:
        print(f"Index {index_name} already exists")

def analyze_text(es, index_name, text):
    """
    Elasticsearch에 텍스트를 분석 요청하는 함수
    """
    analyze_body = {
        "analyzer": "korean_shingle_analyzer",
        "text": text
    }
    response = es.indices.analyze(index=index_name, body=analyze_body)
    return response

def create_es_connection():
    """
    Elasticsearch 연결 설정
    """
    es = Elasticsearch(
        ["https://localhost:9200"],
        basic_auth=("elastic", "S+p+aDSIG=YduplYogRt"),
        verify_certs=False  # SSL 인증서 무시
    )
    return es

def get_index_settings():
    """
    Elasticsearch 인덱스 설정 반환
    """
    settings = {
        "settings": {
            "analysis": {
                "tokenizer": {
                    "standard": {
                        "type": "standard"  # 일반적인 토크나이저 사용
                    }
                },
                "char_filter": {
                    "remove_empty": {
                        "type": "pattern_replace",
                        "pattern": "_+",
                        "replacement": ""
                    }
                },
                "filter": {
                    "my_shingle_filter": {
                        "type": "shingle",
                        "min_shingle_size": 2,
                        "max_shingle_size": 2,
                        "output_unigrams": True,
                        "token_separator": " "  # 언더바 제거
                    }
                },
                "analyzer": {
                    "korean_shingle_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "my_shingle_filter"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "korean_shingle_analyzer"
                }
            }
        }
    }
    return settings