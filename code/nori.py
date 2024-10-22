import json
import re

from elasticsearch import Elasticsearch

stopwords_list =  [
    "은", "는", "이", "가", "을", "를", "에", "의", "도", 
    "그리고", "그래서", "하지만", "그러나", "즉", "또한", "따라서", "더구나", 
    "즉", "결국", "뿐", "따로", "또", "아니면", "그러면", "만약", "하지만", "만일",
    "이렇게", "저렇게", "그렇게", "거기", "여기", "저기", "어떻게", "왜", "무엇",
    "너무", "아주", "매우", "대단히", "좀", "조금", "다시", "정말", "사실", "그냥", 
    "매우", "상당히", "심지어", "아무리", "매번", "마침내", "결코", "가끔", "어쩌면", 
    "또한", "따라서", "그래도", "그럼에도", "때문에", "심지어", "역시", "모두", 
    "결국", "사실은", "결과적으로", "게다가", "예를 들어", "그렇다면", "그렇지 않으면",
    "혹은", "따라서", "이유는", "무엇보다도", "여전히", "또", "대충", "굉장히", 
    "다시", "어쩌면", "정도", "쯤", "말하자면", "말하자면", "왜냐하면", "게다가",
    "이외에", "게다가", "한편", "그동안", "마찬가지로", "결과적으로"
]

def create_index(body=None):
    if not es.indices.exists(index=INDEX):
        return es.indices.create(index=INDEX, body=body)
      
def preprocess_text(text):
    # 특수 문자 제거
    text = re.sub(r'[^가-힣\s]', '', text)

    # 불용어 제거
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords_list] 
    return ' '.join(cleaned_words)

# Elasticsearch에 연결
es = Elasticsearch(
    ["https://localhost:9200"],
    basic_auth=("elastic", "S+p+aDSIG=YduplYogRt"),
    verify_certs=False  # SSL 인증서 무시
) 

INDEX = "bm25_tokenizer" 
# es.indices.delete(index=INDEX)

# Nori Tokenizer + Shingle 필터 적용한 인덱스 생성 설정
settings = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "nori_tokenizer": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed"  # 복합어 분리 설정
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
                # "my_stop_filter": {
                #     "type": "stop",
                #     "stopwords": [
                #         "이", "그", "저", "것", "수", "그리고", "그러나", "또한", "하지만", "즉", "또", 
                #         "의", "가", "을", "를", "에", "에게", "에서", "로", "부터", "까지", "와", 
                #         "과", "도", "은", "는", "이것", "그것", "저것", "뭐", "왜", "어떻게", "어디", "누구", 
                #         "있다", "없다"  # 불용어 리스트 업데이트
                #     ]
                # },
                # "nori_pos_filter": {
                #     "type": "nori_part_of_speech",
                #     "stoptags": [
                #         "E",  # 종결 어미
                #         "IC",  # 감탄사
                #         "J",  # 조사
                #         "MAG",  # 부사 (선택적으로 사용할 수 있음)
                #         "MM",  # 관형사
                #         "SP",  # 문장 기호
                #         "SSC",  # 반점
                #         "SC"  # 구두점
                #     ]
                # },
                "my_shingle_filter": {
                    "type": "shingle",
                    "min_shingle_size": 2,  # 최소 Shingle 크기
                    "max_shingle_size": 3,  # 최대 Shingle 크기
                    "output_unigrams": True,  # 단일 토큰도 출력
                    "token_separator": " "  # 언더바 대신 공백으로 토큰 구분
                }, 
            },
            "analyzer": {
                "korean_shingle_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",  # Nori Tokenizer 사용
                    # "char_filter": ["remove_empty"],
                    "filter": [
                        # "nori_pos_filter",  # POS 필터링 적용
                        # "my_stop_filter",  # 불용어 제거
                        "my_shingle_filter"  # Shingle 필터 적용
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "korean_shingle_analyzer"  # Nori + Shingle Analyzer 적용
            }
        }
    }
}

# 인덱스 생성 요청
if not es.indices.exists(index=INDEX):
    create_index(settings)
    print(f"Index {INDEX} created successfully")
    
# 입력된 텍스트 분석 (Shingle 적용)
def analyze_text(text):
    analyze_body = {
        "analyzer": "korean_shingle_analyzer",
        "text": text
    }
    response = es.indices.analyze(index=INDEX, body=analyze_body)
    return response
  
# 테스트용 텍스트 입력
test_text = "대한민국(한국 한자: 大韓民國)은 동아시아의 한반도 군사 분계선 남부에 위치한 나라이다. 약칭으로 한국(한국 한자: 韓國)과 남한(한국 한자: 南韓, 문화어: 남조선)으로 부르며 현정체제는 대한민국 제6공화국이다. 대한민국의 국기는 대한민국 국기법에 따라 태극기[5]이며, 국가는 관습상 애국가, 국화는 관습상 무궁화이다. 공용어는 한국어와 한국 수어이다. 수도는 서울특별시이다. 인구는 2024년 2월 기준으로 5,130만명[6]이고, 이 중 절반이 넘는(50.74%) 2,603만명이 수도권에 산다.[7]"

# 전처리 후 결과
cleaned_text = preprocess_text(test_text)
print(f"전처리 후 텍스트: {cleaned_text}")

response = analyze_text(cleaned_text)

# 분석 결과 출력
print("Shingle tokens:")
for token in response['tokens']:
    print(token['token'])