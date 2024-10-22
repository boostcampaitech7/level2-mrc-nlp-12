# main.py
from preprocess_nori import preprocess_text
from elasticsearch_bm25 import create_es_connection, create_index, analyze_text, get_index_settings


def create_nori(text):
    INDEX = "bm25_tokenizer"

    # Elasticsearch 연결
    es = create_es_connection()

    # 인덱스 생성
    settings = get_index_settings()
    create_index(es, INDEX, settings)

    # 테스트용 텍스트
    test_text = "대한민국(한국 한자: 大韓民國)은 동아시아의 한반도 군사 분계선 남부에 위치한 나라이다. 약칭으로 한국(한국 한자: 韓國)과 남한(한국 한자: 南韓, 문화어: 남조선)으로 부르며 현정체제는 대한민국 제6공화국이다."

    # 전처리 후 텍스트
    cleaned_text = preprocess_text(test_text)
    print(f"전처리 후 텍스트: {cleaned_text}")

    # 텍스트 분석
    response = analyze_text(es, INDEX, cleaned_text) 
    token_forms = []

    # Shingle 분석 결과 출력
    print("Shingle tokens:")
    for token in response['tokens']:
        token_forms.append(token['token'])
        
    return token_forms