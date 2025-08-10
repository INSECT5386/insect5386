import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import sentencepiece as spm
import requests

# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/chat/resolve/main/NewS3GeN/dataset.jsonl?download=true', 'dataset.jsonl')
download_file('https://huggingface.co/datasets/Yuchan5386/Tokenizer/resolve/main/unigram_model.model?download=true', 'ko_unigram.model')

# ⬇️ JSONL 파일을 Pandas DataFrame으로 읽기
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

df = load_jsonl("dataset.jsonl")
print(f"✅ 데이터 로드 완료: {len(df)}개의 샘플")

# ⬇️ DataFrame에서 질문과 답변 추출하여 <start> q <sep> a <end> 형식으로 변환
def create_qa_sentences(df, max_pairs=50000):
    qa_pairs = []
    for _, row in df.head(max_pairs).iterrows():
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        full = f"<start> {q} <sep> {a} <end>"
        qa_pairs.append(full)
    return qa_pairs

train_sentences = create_qa_sentences(df, max_pairs=50000)
print(f"✅ 전처리 완료: {len(train_sentences)}개의 QA 쌍 생성")

# ⬇️ 토크나이저 불러오기
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

# ⬇️ 특수 토큰 ID 추출
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
sep_id = sp.piece_to_id("<sep>")
end_id = sp.piece_to_id("<end>")
unk_id = sp.piece_to_id("<unk>")

vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

# ⬇️ 텍스트 <-> ID 변환 함수
def text_to_ids(text):
    return sp.encode(text, out_type=int)

def ids_to_text(ids):
    filtered_ids = [id_ for id_ in ids if id_ not in [pad_id, unk_id]]
    return sp.decode(filtered_ids)

# ⬇️ 전처리 하이퍼파라미터
max_len = 256
batch_size = 32

# ⬇️ 인풋과 타겟 마스킹 포함된 전처리
encoded_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    all_ids = text_to_ids(sentence)
    if len(all_ids) >= max_len:
        all_ids = all_ids[:max_len]

    try:
        sep_index = all_ids.index(sep_id)
    except ValueError:
        continue

    input_ids = all_ids[:sep_index + 1]
    target_ids = all_ids[sep_index + 1:]

    full_input = input_ids + target_ids
    full_input = full_input[:max_len]

    target_mask = [0] * len(input_ids) + [1] * len(target_ids)
    target_mask = target_mask[:max_len]

    if len(full_input) < max_len:
        pad_len = max_len - len(full_input)
        full_input += [pad_id] * pad_len
        target_mask += [0] * pad_len

    encoded_inputs.append(full_input)

    target_seq = full_input[1:] + [end_id]
    target_seq = target_seq[:max_len]

    masked_target = [
        t if m == 1 else pad_id
        for t, m in zip(target_seq, target_mask)
    ]
    targets.append(masked_target)

encoded_inputs = np.array(encoded_inputs, dtype=np.int32)
targets = np.array(targets, dtype=np.int32)

print(f"✅ 인코딩 완료: {encoded_inputs.shape[0]} 샘플")

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for input_seq, target_seq in zip(encoded_inputs, targets):
        yield input_seq, target_seq

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ TF Dataset 생성 완료!")

def sharedblock(x):
    skip = x
    d_model = x.shape[-1]
    a = layers.Dense(d_model)(x)
    b = layers.Dense(d_model, activation='sigmoid')(x)
    x = layers.Dense(d_model * 2, activation='gelu')(a)
    return x * b + skip
    
def ModelLM(vocab_size, d_model):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    embedding = layers.Embedding(vocab_size, d_model, mask_zero=True)(inputs)
    block = sharedblock()

    x = block(x)
    x = block(x)
    x = block(x)
    return x


# 손실 및 메트릭 정의
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    masked_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return masked_loss

def masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, preds), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

def masked_perplexity(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    avg_loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    return tf.exp(tf.minimum(avg_loss, 10.0))

def create_lr_schedule(initial_lr=5e-5, decay_steps=10000, decay_rate=0.9):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )

# 모델 생성
model = ModelLM(
    vocab_size=vocab_size,
    d_model=128,
)

# 옵티마이저
optimizer = tf.keras.optimizers.Adam(
    learning_rate=create_lr_schedule(),
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
    clipnorm=1.0
)

# 컴파일
model.compile(
    optimizer=optimizer,
    loss=masked_loss,
    metrics=[masked_accuracy, masked_perplexity]
)

# 더미 입력으로 모델 초기화
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)
model.summary()

# 학습
history = model.fit(
    dataset,
    epochs=1,
    steps_per_epoch=len(encoded_inputs) // batch_size,
    verbose=1
)

# 가중치 저장
model.save_weights("Cobra.weights.h5")
print("✅ 모델 가중치 저장 완료!")

# Google Colab에서 다운로드
try:
    from google.colab import files
    files.download('Cobra.weights.h5')
except ImportError:
    print("Colab이 아닙니다. 파일은 로컬에 저장됨.")

# ======================= 텍스트 생성 함수 =======================
def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    input_text = f"<start> {prompt} <sep>"
    input_ids = text_to_ids(input_text)
    generated = list(input_ids)

    for step in range(max_gen):
        current_len = len(generated)
        if current_len >= max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated

        # 패딩
        padded_input = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([padded_input])  # [1, T]

        # 예측
        logits = model(input_tensor, training=False)  # [1, T, V]
        pos = min(current_len, max_len) - 1
        next_token_logits = logits[0, pos, :].numpy()

        # end, pad에 패널티
        next_token_logits[end_id] -= 5.0
        next_token_logits[pad_id] -= 10.0

        # temperature 적용
        next_token_logits /= temperature
        probs = tf.nn.softmax(next_token_logits).numpy()

        # Top-p (nucleus sampling)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)

        next_token_id = np.random.choice(top_indices, p=top_probs)

        # 최소 길이 이후에만 종료
        if next_token_id == end_id and len(generated) >= min_len:
            break

        generated.append(int(next_token_id))

    # 디코딩 (입력 부분 제외하고 답변만 추출)
    full_text = ids_to_text(generated)
    if "<sep>" in full_text:
        answer = full_text.split("<sep>", 1)[1].strip()
        return answer
    return full_text.strip()

# 테스트 생성
print("\n\n===== 생성 결과 =====")
print(generate_text_topp(model, "안녕하세요", p=0.9, temperature=0.8))
