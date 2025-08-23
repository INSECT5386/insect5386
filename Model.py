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
download_file('https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/Text_Generation/converted.jsonl?download=true', 'dataset.parquet')
download_file('https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/Text_Generation/unigram.model?download=true', 'ko_unigram.model')
model_path = 'Smolwrite-gpt.weights.h5'
# ⬇️ Parquet 데이터 불러오기
df = pd.read_parquet("dataset.parquet", engine="pyarrow")

# ⬇️ <start> 질문 <sep> 답변 <end> 포맷으로 변환
train_sentences = []

for conversations in df["conversations"]:
    for i in range(0, len(conversations) - 1, 2):
        item1, item2 = conversations[i], conversations[i + 1]
        if item1.get("from") == "human" and item2.get("from") == "gpt":
            prompt = item1.get("value", "").strip().replace("\n", " ")
            response = item2.get("value", "").strip().replace("\n", " ")
            full = f"<start> {prompt} <sep> {response} <end>"
            train_sentences.append(full)
train_sentences = train_sentences
print(f"총 문장 개수: {len(train_sentences)}")

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
    return sp.decode(ids)

# ⬇️ 전처리 하이퍼파라미터
max_len = 180
batch_size = 32

# ⬇️ 인풋과 타겟 마스킹 포함된 전처리
encoded_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index + len("<sep>")].strip()
    target_text = sentence[sep_index + len("<sep>"):].strip()

    input_ids = text_to_ids(input_text)
    target_ids = text_to_ids(target_text + " <end>")

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

# ⬇️ 넘파이 변환
encoded_inputs = np.array(encoded_inputs)
targets = np.array(targets)

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

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = layers.LayerNormalization(epsilon=1e-5)
        # 핵심: key_dim = d_model // n_heads  ✅
        self.attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout)
        self.dropout1 = layers.Dropout(dropout)

        self.ln2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff = tf.keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        qkv = self.ln1(x)
        # query=key=value (self-attn) + causal mask ✅
        h = self.attn(query=qkv, value=qkv, key=qkv, use_causal_mask=True, training=training)
        x = x + self.dropout1(h, training=training)
        h = self.ff(self.ln2(x), training=training)
        x = x + self.dropout2(h, training=training)
        return x

class GPT(tf.keras.Model):
    def __init__(self, vocab_size, max_len=160, n_layers=6, d_model=160, n_heads=8, d_ff=640, dropout=0.1, pad_id=3):
        super().__init__()
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb   = layers.Embedding(max_len, d_model)
        self.drop = layers.Dropout(dropout)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)
        self.head = layers.Dense(vocab_size, use_bias=False)
        self.max_len = max_len
        self.pad_id = pad_id

    def call(self, input_ids, training=False):
        T = tf.shape(input_ids)[1]
        pos = tf.range(0, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)   # [B,T,C]
        x = self.drop(x, training=training)
        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.ln_f(x)
        logits = self.head(x)                               # [B,T,V]
        return logits
# 손실 함수 및 메트릭 정의
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
    return tf.exp(tf.minimum(avg_loss, 10.0))  # 수치 안정성 확보

def masked_top5_accuracy(y_true, y_pred):
    top5_preds = tf.nn.top_k(y_pred, k=5).indices
    top5_preds = tf.cast(top5_preds, dtype=y_true.dtype)  # <-- 이 줄 추가
    y_true_expanded = tf.expand_dims(y_true, axis=-1)
    matches = tf.reduce_any(tf.equal(y_true_expanded, top5_preds), axis=-1)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.reduce_sum(mask)

def token_level_loss(y_true, y_pred):
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_mean(loss * mask)

def create_lr_schedule(initial_lr=5e-5, decay_steps=10000, decay_rate=0.9):
    return tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )

cfg = dict(vocab_size=vocab_size, max_len=180, n_layers=6, d_model=160, n_heads=8, d_ff=640, dropout=0.1)
model = GPT(**cfg)
dummy_input = tf.zeros((1, 180), dtype=tf.int32)
_ = model(dummy_input)  # 모델 빌드
model.load_weights(model_path)
print("모델 가중치 로드 완료!")

# 옵티마이저 설정
optimizer = tf.keras.optimizers.Adam(
    learning_rate=create_lr_schedule(),
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
    clipnorm=1.0
)

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss=masked_loss,
    metrics=[
        masked_accuracy,
        masked_perplexity,
        masked_top5_accuracy,
        token_level_loss
    ]
)

# 더미 인풋으로 모델 초기화
dummy_input = np.zeros((1, max_len), dtype=np.int32)
model(dummy_input)
model.summary()

# 학습 시작
history = model.fit(
    dataset,
    epochs=1,
    steps_per_epoch = encoded_inputs.shape[0] // batch_size,
    verbose=1
)

# 가중치 저장
model.save_weights("Cobra.weights.h5")
print("모델 가중치 저장 완료!")


def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    for step in range(max_gen):
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        next_token_logits[end_id] -= 5.0
        next_token_logits[pad_id] -= 10.0
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        next_token_id = np.random.choice(top_indices, p=top_probs)
        if next_token_id == end_id and len(generated) >= min_len:
            break
        generated.append(int(next_token_id))
    return ids_to_text(generated)

print("\n\n===== 생성 결과 =====")  
print(generate_text_topp(model, "안녕", p=0.9))
