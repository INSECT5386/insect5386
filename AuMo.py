import csv
import re
import numpy as np
import tensorflow as tf
import requests
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://raw.githubusercontent.com/INSECT5386/SeQRoN/main/data.csv?spm=a2ty_o01.29997173.0.0.7edec921fx1gz3&file=data.csv', 'MLdata.csv')

# 토크나이저
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# 데이터셋 경로 설정
csv_path = "MLdata.csv"

# CSV에서 (input, output) 쌍 읽기
pairs = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        inp = row["questions"].strip()
        out = row["answers"].strip()
        pairs.append((inp, out))

# 벡터라이저 및 라벨 인코더 준비
vectorizer = CountVectorizer(tokenizer=tokenize, ngram_range=(1, 3))
all_texts = [p[0] + " " + p[1] for p in pairs]
vectorizer.fit(all_texts)

le = LabelEncoder()
le.fit([token for out in [p[1] for p in pairs] for token in tokenize(out)] + ["<EOS>"])

# 데이터 준비 함수 (GPU 호환)
def prepare_data_for_gpu(pairs, vectorizer, label_encoder):
    X_list, y_list = [], []
    for inp, out in pairs:
        tokens = tokenize(out) + ["<EOS>"]
        prefix = []
        for token in tokens:
            context = " ".join(tokenize(inp) + prefix)
            X_list.append(context)
            vec = np.zeros(len(label_encoder.classes_), dtype=np.float32)
            idx = label_encoder.transform([token])[0]
            vec[idx] = 1.0
            y_list.append(vec)
            prefix.append(token)
    X_vec = vectorizer.transform(X_list).astype(np.float32)
    y_vec = np.array(y_list)
    return tf.constant(X_vec.toarray()), tf.constant(y_vec)

# 릿지 회귀 함수 (TensorFlow 기반)
def ridge_regression_tf(X, y, alpha=1.0):
    X = tnp.asarray(X)
    y = tnp.asarray(y)

    if X.shape[1] != y.shape[0]:
        X = tnp.hstack([X, tnp.ones((X.shape[0], 1))])

    XtX = X.T @ X
    I = tnp.eye(XtX.shape[0], dtype=X.dtype)
    XtX_reg = XtX + alpha * I

    Xty = X.T @ y
    w = tnp.linalg.solve(XtX_reg, Xty)  # ❗ inv() 대신 solve()
    return w

# 앙상블 모델 학습
n_models = 3
ridge_weights_list = []

X_train_tensor, y_train_tensor = prepare_data_for_gpu(pairs[:250], vectorizer, le)

for i in range(n_models):
    print(f"[INFO] 모델 {i+1} 학습 중...")
    noise = tf.random.normal(shape=tf.shape(X_train_tensor), stddev=0.01, dtype=tf.float32)
    X_noisy = tf.clip_by_value(X_train_tensor + noise, 0.0, tf.reduce_max(X_train_tensor))
    weights = ridge_regression_tf(X_noisy, y_train_tensor, alpha=1.0)
    ridge_weights_list.append(weights)

# Top-P 샘플링 (GPU 지원)
def top_p_sampling_gpu(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = tf.nn.softmax(logits)
    sorted_indices = tf.argsort(probs, direction='DESCENDING')[0]
    sorted_probs = tf.gather(probs[0], sorted_indices)
    cumulative_probs = tf.math.cumsum(sorted_probs)

    cutoff = tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32)) + 1
    candidates = sorted_indices[:cutoff]
    candidate_probs = sorted_probs[:cutoff]
    candidate_probs = candidate_probs / tf.reduce_sum(candidate_probs)

    chosen_index = tf.random.categorical(tf.math.log([candidate_probs]), 1)[0, 0]
    chosen_token_idx = candidates[chosen_index]
    return chosen_token_idx.numpy().item()

# 생성 함수
def generate(model_weights_list, input_text, max_len=20, top_p=0.9, temperature=1.0):
    generated = []
    print(f"[DEBUG] 입력 문장: {input_text}")
    for step in range(max_len):
        context = " ".join(tokenize(input_text) + generated)
        
        # CountVectorizer → toarray() 추가!
        X_context = vectorizer.transform([context]).astype(np.float32)
        X_tensor = tf.constant(X_context.toarray())  # ← 여기가 핵심!

        preds = []
        for w in model_weights_list:
            pred = tf.matmul(X_tensor, w)
            preds.append(pred)

        avg_pred = tf.reduce_mean(preds, axis=0)
        sims = tf.nn.softmax(avg_pred)

        next_idx = top_p_sampling_gpu(sims, p=top_p, temperature=temperature)
        next_token = le.inverse_transform([next_idx])[0]

        print(f"[DEBUG] step {step} - next_token: {next_token}")

        if next_token == "<EOS>":
            print("[DEBUG] <EOS> 토큰 발견, 종료!")
            break
        generated.append(next_token)

    output = " ".join(generated)
    print(f"[DEBUG] 생성된 문장: {output}")
    return output

# 테스트 문장들
print(generate(ridge_weights_list, "안녕"))
print(generate(ridge_weights_list, "오늘 날씨 어때?"))
print(generate(ridge_weights_list, "이름 뭐야?"))
