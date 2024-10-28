# preprocess.py
import re

stopwords_list = [
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

def preprocess_text(text):
    """
    특수 문자를 제거하고 불용어를 제거하는 전처리 함수
    """
    # 특수 문자 제거
    text = re.sub(r'[^가-힣\s]', '', text)

    # 불용어 제거
    words = text.split()
    cleaned_words = [word for word in words if word not in stopwords_list]
    
    return ' '.join(cleaned_words)