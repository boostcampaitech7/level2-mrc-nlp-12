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

def create_es_connection(INDEX):
    """
    Elasticsearch 연결 설정
    """
    es = Elasticsearch(
        ["https://localhost:9200"],
        basic_auth=("elastic", "S+p+aDSIG=YduplYogRt"),
        verify_certs=False  # SSL 인증서 무시
    )
    
    if es.indices.exists(index=INDEX):
        es.indices.delete(index=INDEX)
        print(f"Index {INDEX} deleted successfully.")
    else:
        print(f"Index {INDEX} does not exist.")
    
    return es

def get_index_settings():
    """
    Elasticsearch 인덱스 설정 반환
    """
    settings = { 
        "settings": { 
            "analysis": {
                "tokenizer": {
                    "nori_tokenizer": {
                        "type": "nori_tokenizer", 
                        "decompound_mode": "discard"
                    }
                },
                "filter": {
                    "nori_part_of_speech_filter": {
                        "type": "nori_part_of_speech",
                        "stoptags": [  # 제외할 품사 태그 목록
                            "E",  # 어미
                            "IC",  # 감탄사
                            "J",  # 조사
                            "MAG",  # 일반 부사
                            "MAJ",  # 접속 부사
                            "MM",  # 관형사
                            "SP",  # 문장 부호
                            "SSC", "SSO",  # 괄호
                            "SC",  # 구두점
                            "SE",  # 줄임표
                            "XPN",  # 접두사
                            "XSA",  # 형용사 파생 접미사
                            "XSN",  # 명사 파생 접미사
                            "XSV"  # 동사 파생 접미사
                        ]
                    },
                    "shingle_filter": {
                        "type": "shingle",
                        "min_shingle_size": 2,
                        "max_shingle_size": 2,
                        "output_unigrams": True
                    }, 
                    "korean_stop": {
                        "type": "stop",
                        "stopwords": ["은", "는", "이", "가", "을", "를", "에", "의", "그", "인", '않', "되", "안",], 
                        "enable_position_increments": "false"
                    }
                },
                "analyzer": {
                    "korean_shingle_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                        "filter": [
                            "nori_part_of_speech_filter",  # 품사 필터 적용
                            "lowercase",
                            "shingle_filter", 
                            "korean_stop"
                        ]
                    }
                }
            }, 
            "index": {
                "analyze.max_token_count": 40000
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "korean_shingle_analyzer",
                    "search_analyzer": "korean_shingle_analyzer" 
                }
            }
        }
    }
    return settings