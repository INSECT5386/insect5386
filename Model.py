import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
from tensorflow.keras.initializers import RandomNormal
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
train_sentences = train_sentences[:100000] # 예제용 소량
print(f"총 문장 개수: {len(train_sentences)}")

# ⬇️ 토크나이저 불러오기
sp = spm.SentencePieceProcessor()
sp.load("ko_unigram.model")

# ⬇️ 특수 토큰 ID 추출
pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0  
start_id = sp.piece_to_id("<start>")  
sep_id = sp.piece_to_id("<sep>")  
end_id = sp.piece_to_id("<end>")  

vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

# ⬇️ 전처리 하이퍼파라미터
max_enc_len = 128 # 인코더 최대 길이 (질문 부분)
max_dec_len = 128 # 디코더 최대 길이 (답변 부분)
batch_size = 64

# ⬇️ 전처리 결과 저장할 리스트
encoder_inputs = []
decoder_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index].strip() # 질문 부분
    target_text = sentence[sep_index + len("<sep>"):].strip() # 답변 부분

    # 인코더 입력: 질문 + <sep>
    enc_ids = sp.encode(input_text + " <sep>")[:max_enc_len]

# 디코더 입력: <start> + 답변[:-1]
    dec_input_ids = [start_id] + sp.encode(target_text)[:max_dec_len - 2]

# 정답 라벨: 답변 + <end>
    target_ids = sp.encode(target_text)[:max_dec_len - 1] + [end_id]
    # 패딩 추가
    enc_padded = enc_ids + [pad_id] * (max_enc_len - len(enc_ids))
    dec_padded = dec_input_ids + [pad_id] * (max_dec_len - len(dec_input_ids))
    target_padded = target_ids + [pad_id] * (max_dec_len - len(target_ids))

    encoder_inputs.append(enc_padded)
    decoder_inputs.append(dec_padded)
    targets.append(target_padded)

# ⬇️ 넘파이 배열로 변환
encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
targets = np.array(targets, dtype=np.int32)

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for enc, dec, tgt in zip(encoder_inputs, decoder_inputs, targets):
        # 딕셔너리 대신 튜플 형태로 반환
        yield (enc, dec), tgt

output_types = (
    (tf.int32, tf.int32), # 두 개의 입력 텐서에 대한 타입
    tf.int32 # 타겟에 대한 타입
)

output_shapes = (
    (tf.TensorShape([max_enc_len]), tf.TensorShape([max_dec_len])), # 두 개의 입력 텐서에 대한 모양
    tf.TensorShape([max_dec_len]) # 타겟에 대한 모양
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=output_types,
    output_shapes=output_shapes
)

dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("dataset ok")


# ===== 1. 가변 위치 인코딩 =====
class LearnablePositionalEmbedding(layers.Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.add = layers.Add()
        pos_emb = RandomNormal()(shape=[max_length, d_model])
        self.pos_emb = tf.Variable(
            initial_value=pos_emb,
            trainable=True,
            name='positional_embedding'
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return self.add([inputs, self.pos_emb[tf.newaxis, :seq_len, :]])

class SeProdBlock(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(dim)
        self.norm1 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout_rate)
        self.add = layers.Add()

        self.dense2 = layers.Dense(dim * 2)  # GLU 스타일로 나눌 준비
        self.dense3 = layers.Dense(dim)
        self.norm2 = layers.LayerNormalization()
        self.multi = layers.Multiply()
        self.multi1 = layers.Multiply()
        self.multi2 = layers.Multiply()

    def call(self, x, training=None):
        batch_size, seq_len, d_model = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # ===== Forward Block =====
        A = self.dense1(x)
        B = tf.sigmoid(A)
        C = self.multi([x, B])
        D = self.norm1(C)
        E = tf.nn.silu(D)
        forward_out = self.add([E, x])

        # ===== Reverse Block (GLU Style) =====
        A2 = self.dense2(x)  # [batch, seq, d_model * 2]
        a, b = tf.split(A2, num_or_size_splits=2, axis=-1)
        a = tf.sigmoid(a)
        reverse_out = self.multi1([a, b])
        reverse_out = self.norm2(reverse_out)
        reverse_out = tf.nn.silu(reverse_out)

        # ===== Merge Output =====
        combined = self.multi2([forward_out, reverse_out])  
        combined = self.dense3(combined)
        return combined


# ===== 모델 구성 =====
# 인코더 경로
encoder_input = Input(shape=(max_enc_len,), name='encoder_input')
x_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(encoder_input)
x_pos = LearnablePositionalEmbedding(max_enc_len, d_model)(x_emb)

context_vector = SeProdBlock(d_model, dropout_rate=dropout_rate)(x_pos, training=True)

# 디코더 경로
decoder_input = Input(shape=(max_dec_len,), name='decoder_input')
y_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(decoder_input)
y_pos = LearnablePositionalEmbedding(max_dec_len, d_model)(y_emb)
decoder_output = SeProdBlock(d_model, dropout_rate=dropout_rate)(y_pos, training=True)

cross = layers.Multiply()([decoder_output, context_vector])
cross = layers.LayerNormalization()(cross)
output = SeProdBlock(d_model, dropout_rate=dropout_rate)(cross, training=True)

# 최종 출력
logits = layers.Dense(vocab_size)(output)

model = Model(inputs=[encoder_input, decoder_input], outputs=logits, name='SeProd')

# ===== 컴파일 및 학습 =====
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 모델 요약
model.summary()

model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)


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
model = SeProd(
    vocab_size=vocab_size,
    d_model=256,
    n_layers=6,
    max_len=256
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
