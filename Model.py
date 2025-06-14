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
dataset = dataset.repeat()

print("✅ TF Dataset 생성 완료!")

class SwiGLUFFN(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense = layers.Dense(hidden_dim * 2, use_bias=True)
        self.output_proj = layers.Dense(hidden_dim, use_bias=True)

    def call(self, x):
        # 병렬 연산으로 projection 한 번에
        x_proj = self.dense(x)  # (batch, seq, hidden_dim*2)
        x1, x2 = tf.split(x_proj, num_or_size_splits=2, axis=-1)
        x = tf.nn.silu(x1) * x2  # SwiGLU activation
        return self.output_proj(x)

class S4Core(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        self.A_real = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='A_real')
        self.A_imag = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='A_imag')

        self.B_real = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='B_real')
        self.B_imag = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='B_imag')

        self.C_real = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='C_real')
        self.C_imag = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True, name='C_imag')

        self.D = self.add_weight(shape=(d_model,), initializer='zeros', dtype=tf.float32, trainable=True)

    def call(self, u):
        u_orig = u
        batch = tf.shape(u)[0]
        seq_len = tf.shape(u)[1]
        d_model = tf.shape(u)[2]

        # causal FFT 길이 및 padding
        fft_len_float = tf.math.log(tf.cast(2 * seq_len - 1, tf.float32)) / tf.math.log(2.0)
        fft_len = tf.cast(tf.math.pow(2.0, tf.math.ceil(fft_len_float)), tf.int32)
        pad_len = fft_len - seq_len

        # 시간축
        t = tf.cast(tf.range(seq_len), tf.complex64)  # (seq_len,)

        # 복소수 파라미터
        A_c = tf.complex(self.A_real, self.A_imag)
        B_c = tf.complex(self.B_real, self.B_imag)
        C_c = tf.complex(self.C_real, self.C_imag)

        # A^t
        A_t = tf.pow(tf.expand_dims(A_c, 0), tf.expand_dims(t, 1))  # (seq_len, d_model)
        kernel = tf.expand_dims(C_c, 0) * A_t * tf.expand_dims(B_c, 0)  # (seq_len, d_model)
        kernel = tf.transpose(kernel, [1, 0])  # (d_model, seq_len)

        # causal zero padding (right-side only)
        kernel = tf.pad(kernel, [[0, 0], [0, pad_len]])  # (d_model, fft_len)

        kernel_fft = tf.signal.fft(kernel)  # (d_model, fft_len)

        # 입력 padding 및 FFT
        u_t = tf.transpose(u, [0, 2, 1])  # (batch, d_model, seq_len)
        u_padded = tf.pad(u_t, [[0, 0], [0, 0], [0, pad_len]])  # (batch, d_model, fft_len)
        U_f = tf.signal.fft(tf.cast(u_padded, tf.complex64))

        # pointwise 곱셈 후 IFFT
        Y_f = U_f * tf.expand_dims(kernel_fft, 0)  # (batch, d_model, fft_len)
        y_full = tf.signal.ifft(Y_f)[..., :seq_len]  # (batch, d_model, seq_len)
        y = tf.math.real(y_full)
        y = tf.transpose(y, [0, 2, 1])  # (batch, seq_len, d_model)

        return y + self.D[None, None, :] * u_orig


class Cobrablock(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.core = S4Core(d_model, seq_len=seq_len)
        self.dropout1 = layers.Dropout(dropout_rate)
        
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn = SwiGLUFFN(d_model)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # PreNorm + Residual (Core)
        x_norm = self.norm1(x)
        x_core = self.core(x_norm)
        x = x + self.dropout1(x_core, training=training)

        # PreNorm + Residual (FFN)
        x_norm = self.norm2(x)
        x_ffn = self.ffn(x_norm)
        x = x + self.dropout2(x_ffn, training=training)

        return x
        
# ======================= CobraModel ======================
class CobraModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        self.blocks = [Cobrablock(d_model, seq_len=100, dropout_rate=dropout_rate) for _ in range(n_layers)]
        self.ln_f = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        x = self.token_embedding(x)

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
    d_model=192,
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
