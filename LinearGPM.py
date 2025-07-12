import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multioutput import MultiOutputRegressor
import random

# 토크나이저
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# 데이터 준비: (context, one-hot 벡터) 생성
def prepare_data(pairs, label_encoder, contexts_only=False):
    X, y_vecs = [], []
    for inp, out in pairs:
        inp_tokens = tokenize(inp)
        out_tokens = tokenize(out) + ["<EOS>"]
        prefix = []
        for token in out_tokens:
            context = " ".join(inp_tokens + prefix)
            X.append(context)
            if not contexts_only:
                vec = np.zeros(len(label_encoder.classes_), dtype=float)
                idx = label_encoder.transform([token])[0]
                vec[idx] = 1.0
                y_vecs.append(vec)
            prefix.append(token)
    if contexts_only:
        return X
    return X, np.vstack(y_vecs)

# Top-P 샘플링 함수
def top_p_sampling(probabilities, p=0.9):
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, p) + 1
    candidates = sorted_indices[:cutoff]
    candidate_probs = sorted_probs[:cutoff]
    candidate_probs /= candidate_probs.sum()
    chosen = np.random.choice(candidates, p=candidate_probs)
    return chosen

class AuMoRegressionEnsemble:
    def __init__(self, n_models=3):
        self.vectorizer = CountVectorizer(ngram_range=(1,3))
        self.le = LabelEncoder()
        self.models = [MultiOutputRegressor(Ridge(alpha=1.0)) for _ in range(n_models)]
        self.vocab_embeddings = None

    def fit(self, pairs):
        tokens = []
        for _, out in pairs:
            tokens.extend(tokenize(out))
        tokens.append("<EOS>")
        self.le.fit(tokens)

        X_text, y_vecs = prepare_data(pairs, self.le)
        X_vec = self.vectorizer.fit_transform(X_text)

        # vocab_embeddings는 원-핫 단위행렬로!
        self.vocab_embeddings = np.eye(len(self.le.classes_))

        for i, model in enumerate(self.models):
            print(f"[INFO] 모델 {i+1} 학습 중...")
            model.fit(X_vec, y_vecs)
        print("[INFO] 앙상블 학습 완료!")

    def generate(self, input_text, max_len=20, top_p=0.9):
        generated = []
        print(f"[DEBUG] 입력 문장: {input_text}")
        for step in range(max_len):
            context = " ".join(tokenize(input_text) + generated)
            X_vec = self.vectorizer.transform([context])

            preds = [model.predict(X_vec)[0] for model in self.models]
            avg_pred = np.mean(preds, axis=0)

        # 코사인 유사도 대신 닷 프로덕트 사용
            sims = np.dot(self.vocab_embeddings, avg_pred)

            next_idx = top_p_sampling(sims, p=top_p)
            next_token = self.le.inverse_transform([next_idx])[0]

            print(f"[DEBUG] step {step} - next_token: {next_token} (sim: {sims[next_idx]:.4f})")

            if next_token == "<EOS>":
                print("[DEBUG] <EOS> 토큰 발견, 종료!")
                break
            generated.append(next_token)

        output = " ".join(generated)
        print(f"[DEBUG] 생성된 문장: {output}")
        return output



if __name__ == "__main__":
    # 데이터셋 경로와 읽기
    import csv

    csv_path = "MLdata.csv"  # 여기에 네 데이터셋 경로 넣어라

    # CSV에서 (input, output) 쌍 읽기
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row["input_text"].strip()
            out = row["output_text"].strip()
            pairs.append((inp, out))

    # 모델 학습 + 생성 테스트
    aumodel = AuMoRegressionEnsemble()
    aumodel.fit(pairs[:100])

    # 테스트 문장들
    print(aumodel.generate("안녕하세요"))
    print(aumodel.generate("오늘 날씨 어때?"))
    print(aumodel.generate("이름 뭐야?"))
