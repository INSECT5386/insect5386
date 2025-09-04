import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
import json
import time
import re
import random
import os
import faiss
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# === Custom layers ===
class L2NormLayer(layers.Layer):
    def __init__(self, axis=1, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)
    def get_config(self):
        return {"axis": self.axis, "epsilon": self.epsilon, **super().get_config()}

class LearnableWeightedPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = None  # 초기엔 None으로 선언해둠

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, embed_dim)
        self.dense = layers.Dense(1, use_bias=False)
        self.dense.build(input_shape)  # Dense 레이어 build 호출해서 가중치 생성
        self.built = True  # build 완료 플래그 설정

    def call(self, inputs, mask=None):
        scores = self.dense(inputs)  # (batch, seq_len, 1)

        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            minus_inf = -1e9
            scores = scores + (1 - mask[..., tf.newaxis]) * minus_inf

        weights = tf.nn.softmax(scores, axis=1)  # (batch, seq_len, 1)
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # (batch, embed_dim)
        return weighted_sum
    

# === 환경 변수 및 토큰 ===
os.environ["HF_HOME"] = "/tmp/hf_cache"
hf_token = os.getenv("HF_TOKEN")

# === 상수 ===
MAX_SEQ_LEN = 256
BATCH_SIZE = 1024
SIM_THRESHOLD = 0.0  # 필요에 따라 조정 가능

# === 모델, 토크나이저, 데이터 다운로드 및 로드 ===
TK_MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM-4", filename="ko_bpe.json", repo_type="model", token=hf_token)
MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM-4", filename="sentence_encoder_model.keras", repo_type="model", token=hf_token)
JSONL_PATH = hf_hub_download(repo_id="Yuchan5386/Kode-2", filename="filtered_conversations.jsonl", repo_type="dataset", token=hf_token)
EMBEDDINGS_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM-4", filename="answer_embeddings_streaming.npz", repo_type="model", token=hf_token)
INDEX_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM-4", filename="answer_faiss_index.index", repo_type="model", token=hf_token)

# tokenizers 라이브러리 토크나이저 로드
tokenizer = Tokenizer.from_file(TK_MODEL_PATH)
# === 인코더 로드 ===
encoder = load_model(MODEL_PATH, custom_objects={"L2NormLayer": L2NormLayer, "LearnableWeightedPooling": LearnableWeightedPooling})

# === 답변 텍스트 로드 ===
def load_answers_from_jsonl(jsonl_path):
    answers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                for turn in convs:
                    if turn.get("from") == "gpt" and "value" in turn:
                        answers.append(turn["value"])
            except json.JSONDecodeError:
                continue
    return answers

answer_texts = load_answers_from_jsonl(JSONL_PATH)

# === 임베딩 로드 ===
embedding_npz = np.load(EMBEDDINGS_PATH)
answer_embs = embedding_npz['embeddings'].astype('float32')

# === faiss 인덱스 초기화 및 로드/생성 ===
dim = answer_embs.shape[1]
if os.path.isfile(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatIP(dim)      # FAISS 인덱스를 faiss_index 변수에 담습니다.
    faiss_index.add(answer_embs)
    faiss.write_index(faiss_index, INDEX_PATH)


def tk_tokenize(texts):
    # texts가 단일 문자열이면 리스트로 감싸기
    if isinstance(texts, str):
        texts = [texts]
    encoded = [tokenizer.encode(text).ids for text in texts]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=MAX_SEQ_LEN, padding='post', truncating='post'
    )
    return padded


# === 임베딩 생성 ===
def encode_sentences(texts):
    seqs = tk_tokenize(texts)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0).astype('float32')

def remove_invalid_unicode(text):
    return re.sub(r'[\ud800-\udfff]', '', text)



FILTERED_SENTENCES = [
    "sklearn.preprocessing에서 StandardScaler를 가져옵니다.sklearn.decomposition에서 PCA를 가져옵니다.파이프라인 = 파이프라인(단계=[    ('스케일러', StandardScaler()),    ('pca', PCA())])",
    """def create_sentence(words):    sentence = ""    missing_words = []    for word in words:        if word.endswith("."):            sentence += word        else:            sentence += word + " "    if len(missing_words) > 0:        print("다음 단어가 누락되었습니다: ")        for word in missing_words:            missing_word = input(f"{word}를 입력하세요: ")            sentence = sentence.replace(f"{word} ", f"{missing_word} ")    문장을 반환합니다.이 함수를 사용하는 방법은 다음과 같습니다:```pythonwords = ["이것", "이것", ... ]"""
]

def filter_response(text):
    for f_text in FILTERED_SENTENCES:
        if f_text in text:
            return ""
    return text

QUERY_EXPANSION_DICT = {
    "tensorflow": "import tensorflow as tf",
    "keras": "import tensorflow.keras as keras",
    "numpy": "import numpy as np",
    "pandas": "import pandas as pd",
    "sklearn": "import sklearn",
    "knn": "sklearn knn knn knn Knn KNN",
    "KNN": "sklearn knn knn knn Knn KNN"
}

def expand_query(text):
    return QUERY_EXPANSION_DICT.get(text.strip().lower(), text)

def generate_response(user_input, top_k=3, similarity_threshold=SIM_THRESHOLD):
    user_input = expand_query(user_input)
    augmented = user_input
    query_emb = encode_sentences([augmented])
    D, I = faiss_index.search(query_emb, top_k)  # index → faiss_index 로 변경
    sims = D[0]
    best_idx = I[0][0]
    best_score = sims[0]

    if best_score < similarity_threshold:
        return random.choice([
            "음... 이건 그냥 넘길게요!", "그건 잘 모르겠어요. 😅",
            "이건 내 지식 밖인 것 같아요.", "패스! 다른 질문 가보자고~"
        ])

    best_response = answer_texts[best_idx]

    return filter_response(best_response)