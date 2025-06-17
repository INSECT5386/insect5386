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
train_sentences = train_sentences[:300]
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
max_len = 256
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


class SimpleFFN(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = layers.Dense(dim)
        self.up_proj = layers.Dense(dim)
        self.down_proj = layers.Dense(dim)

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# ==================== RealMambaCore =====================
class RealMambaCore(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.ffn = SimpleFFN(hidden_dim)  # 또는 'tanh'

        self.gate_proj = layers.Dense(hidden_dim)
        self.input_proj = layers.Dense(hidden_dim)

        self.A = self.add_weight(shape=(hidden_dim,),
                                 initializer=tf.keras.initializers.RandomNormal(mean=-0.5, stddev=0.1),
                                 trainable=True, name="A")
        self.B = self.add_weight(shape=(hidden_dim,),
                                 initializer='random_normal',
                                 trainable=True, name="B")
        self.C = self.add_weight(shape=(hidden_dim,),
                                 initializer='random_normal',
                                 trainable=True, name="C")
        self.D = self.add_weight(shape=(hidden_dim,),
                                 initializer='zeros',
                                 trainable=True, name="D")

        self.norm = layers.LayerNormalization()
        self.output_proj = layers.Dense(hidden_dim)

    def fft_convolve(self, u_t, kernel_t, T):
        pad_len = T - 1
        seq_len = T + pad_len

        fft_len_float = tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32)) / tf.math.log(2.0))
        fft_len = tf.cast(2 ** fft_len_float, tf.int32)

        u_padded = tf.pad(u_t, [[0, 0], [0, 0], [pad_len, fft_len - seq_len]])
        K_padded = tf.pad(kernel_t, [[0, 0], [0, fft_len - T]])

        U_f = tf.signal.fft(tf.cast(tf.complex(u_padded, 0.0), tf.complex64))
        K_f = tf.signal.fft(tf.cast(tf.complex(K_padded, 0.0), tf.complex64))

        Y_f = U_f * tf.expand_dims(K_f, 0)
        y_full = tf.signal.ifft(Y_f)
        y_real = tf.math.real(y_full)[..., pad_len:pad_len + T]

        return y_real

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = self.hidden_dim

        gate = tf.nn.silu(self.gate_proj(x))
        x_proj = self.input_proj(x)
        u = gate * x_proj

        time_idx = tf.cast(tf.range(T), dtype=self.A.dtype)[:, None]
        A_pow = tf.pow(tf.expand_dims(self.A, 0), time_idx)
        kernel = self.B * A_pow

        u_t = tf.transpose(u, [0, 2, 1])
        kernel_t = tf.transpose(kernel, [1, 0])

        y_real = self.fft_convolve(u_t, kernel_t, T)
        y = tf.transpose(y_real, [0, 2, 1])

        y = self.C * y + self.D * u

        y = self.norm(y)
        y = self.ffn(y)
        y = self.output_proj(y)

        return y

# ======================= Cobrablock ======================
class Cobrablock(tf.keras.layers.Layer):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mamba = RealMambaCore(d_model)
        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = residual + self.dropout1(x, training=training)

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

def advanced_generate_text(
    model, 
    prompt, 
    max_len=256, 
    max_gen=200, 
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    min_len=20
):
    """
    고급 텍스트 생성 함수를 generate_text_topp와 유사하게 간소화한 버전이며,
    각 생성된 토큰을 yield를 통해 순차적으로 반환합니다.
    """
    
    # 초기 입력 처리
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    
    # 최초의 생성된 토큰 (프롬프트 + 시작 토큰)은 yield하지 않음 (필요시 추가 가능)
    for step in range(max_gen):
        # 입력 시퀀스 관리
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated

        input_padded = np.pad(
            input_seq, 
            (0, max_len - len(input_seq)), 
            constant_values=pad_id
        )
        input_tensor = tf.convert_to_tensor([input_padded])
        
        # 예측 수행
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        
        # 반복 페널티 적용
        for i, token_id in enumerate(generated[-50:]):  # 최근 50개 토큰에 페널티
            if token_id < len(next_token_logits):
                next_token_logits[token_id] /= repetition_penalty
        
        # 특수 토큰 보정
        next_token_logits[pad_id] -= 10.0
        if len(generated) < min_len:
            next_token_logits[end_id] -= 5.0
            
        # 온도 적용 및 Top-p 샘플링
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff = np.searchsorted(cumulative_probs, top_p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        
        # 확률 정규화
        top_probs /= np.sum(top_probs)
        
        # 다음 토큰 선택
        next_token_id = np.random.choice(top_indices, p=top_probs)
        
        # 종료 조건 체크
        if next_token_id == end_id and len(generated) >= min_len:
            break
        
        # 토큰 추가 및 yield
        generated.append(int(next_token_id))
        yield ids_to_text([next_token_id])  # 하나의 토큰만 반환

    # 전체 결과는 필요시 별도로 반환하거나 외부에서 모으도록 함

def decode_sp_tokens(tokens):
    """
    SentencePiece 기반의 토큰 리스트를 사람이 읽을 수 있는 텍스트로 디코딩합니다.
    '▁' (underscore)는 공백으로 대체하고, 전체 문자열은 양쪽 공백을 제거합니다.
    
    Args:
        tokens (list of str): 각 요소가 하나의 토큰인 리스트
        
    Returns:
        str: 디코딩된 텍스트
    """
    text = ''.join(tokens).replace('▁', ' ').strip()
    return text
def generate_full_text(model, prompt, decode_fn=None, **kwargs):
    """
    advanced_generate_text_yield 제너레이터를 소비하여
    전체 텍스트 응답을 반환합니다.
    
    Args:
        model: 텍스트 생성 모델
        prompt (str): 입력 프롬프트
        decode_fn (function): 토큰 디코딩 함수 (예: decode_sp_tokens)
        **kwargs: advanced_generate_text_yield에 전달할 추가 파라미터
    
    Returns:
        str: 생성된 전체 텍스트
    """
    generator = advanced_generate_text(model, prompt, **kwargs)
    
    tokens = []
    for token in generator:
        tokens.append(token)
    
    if decode_fn:
        return decode_fn(tokens)
    else:
        return ''.join(tokens)

# 예시 프롬프트
prompt = "안녕하세요"

# 전체 텍스트 생성
response = generate_full_text(
    model,
    prompt,
    decode_fn=decode_sp_tokens,
    max_len=256,
    max_gen=200,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    min_len=20
)

print(f"프롬프트: {prompt}")
print(f"응답: {response}")
