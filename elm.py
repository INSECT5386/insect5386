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
download_file('https://huggingface.co/datasets/Yuchan5386/KeraLux4/resolve/main/dataset.parquet?download=true', 'dataset.parquet')
download_file('https://huggingface.co/datasets/Yuchan5386/KeraLux4/resolve/main/kolig_unigram.model?download=true', 'ko_unigram.model')

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
train_sentences = train_sentences[:50]
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
max_len = 100
batch_size = 80

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

class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = layers.Dense(dim)
        self.up_proj = layers.Dense(dim)
        self.down_proj = layers.Dense(dim)

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ======================= RoPE ==========================
def apply_rope(x):
    seq_len = tf.shape(x)[1]
    dim = tf.shape(x)[2]
    half_dim = dim // 2

    position = tf.cast(tf.range(seq_len), tf.float32)  # (T,)  <-- 이게 핵심!
    freq = tf.pow(10000.0, -tf.range(half_dim, dtype=tf.float32) / tf.cast(half_dim, tf.float32))  # (D/2,)
    angles = tf.einsum('i,j->ij', position, freq)  # (T, D/2)

    sin = tf.sin(angles)[None, :, :]  # (1, T, D/2)
    cos = tf.cos(angles)[None, :, :]  # (1, T, D/2)

    x1 = x[:, :, :half_dim]
    x2 = x[:, :, half_dim:]
    x_rot = tf.concat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return x_rot

# ======================= Cobrablock ======================
class Cobrablock(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = Layers.MultiHeadAttention(8, 32)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.ffn = SwiGLUFFN(d_model)
    def call(self, x, training=False):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = residual + self.dropout1(x, training=training)

        x = self.ffn(x)

        residual = x
        x = self.norm2(x)
        x = self.dropout2(x, training=training)
        x = residual + x

        return x

# ======================= CobraModel ======================
class CobraModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.blocks = [Cobrablock(d_model, dropout_rate) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        x = self.token_embedding(x)
        x = apply_rope(x)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.ln_f(x)
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)
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


# 모델 생성
model = CobraModel(
    vocab_size=vocab_size,
    d_model=256,
    n_layers=10
)

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
from google.colab import files
files.download('Cobra.weights.h5')  # 여기에 다운로드할 파일명을 넣어줘


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

