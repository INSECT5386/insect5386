
import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
import sentencepiece as spm  
import requests
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import RandomNormal


# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/dataaaa/resolve/main/dataset.parquet?download=true', 'dataset.parquet')
download_file('https://huggingface.co/datasets/Yuchan5386/dataaaa/resolve/main/kolig_unigram.model?download=true', 'ko_unigram.model')

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

class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_loss = tf.Variable(0.0, dtype=tf.float32)
        self.total_count = tf.Variable(0, dtype=tf.int64)
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = self.loss_obj(y_true, y_pred)

        # batch_size와 seq_len을 int64로 명시 변환
        batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)
        seq_len = tf.cast(tf.shape(y_true)[1], tf.int64)

        self.total_loss.assign_add(loss)
        self.total_count.assign_add(batch_size * seq_len)

    def result(self):
        avg_log_likelihood = self.total_loss / tf.cast(self.total_count, tf.float32)
        return tf.exp(avg_log_likelihood)

    def reset_states(self):
        self.total_loss.assign(0.0)
        self.total_count.assign(0)


class LearnablePositionalEmbedding(layers.Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            shape=(self.max_length, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True,
            name='positional_embedding'
        )
        self.add = layers.Add()

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        x = self.pos_emb[tf.newaxis, :seq_len, :]
        return self.add([inputs, x])


class Core(layers.Layer):
    def __init__(self, dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.norm = layers.LayerNormalization()
        self.W = layers.Dense(self.dim * 2)
        self.W1 = layers.Dense(self.dim * 2)
        self.W2 = layers.Dense(self.dim)
        self.W3 = layers.Dense(self.dim)
        self.dropout = layers.Dropout(self.dropout_rate)
        self.add_layer = layers.Add()
        self.multiply = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.W2(inputs)             # (batch, seq_len, dim)
        x = self.W(x)                  # (batch, seq_len, dim*2)
        x_S = tf.sigmoid(x)             # gating 값
        x = self.multiply([x, x_S])     # element-wise gating
        x = self.W3(x)                 # (batch, seq_len, dim)
        
        a, b = tf.split(x, 2, axis=-1) # SwiGLU 1
        a = tf.nn.gelu(a)
        x = self.multiply([a, b])
        
        x = self.W1(x)                 # (batch, seq_len, dim*2)
        a, b = tf.split(x, 2, axis=-1) # SwiGLU 2
        a = tf.nn.gelu(a)
        x = self.multiply([a, b])
        
        x = self.dropout(x, training=training)
        x = self.add_layer([x, inputs]) # residual
        return self.norm(x)


class LinearFWLayer(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        self.Wq = layers.Dense(self.dim)
        self.Wk = layers.Dense(self.dim)
        self.Wv = layers.Dense(self.dim)
        self.o = layers.Dense(self.dim)
        self.multiply = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs, context):
        q = self.Wq(inputs)    # (B, Lq, D)
        k = self.Wk(context)   # (B, Lk, D)
        v = self.Wv(context)   # (B, Lk, D)

        # φ 함수: elu + 1 같은 간단한 커널 함수
        phi = lambda x: tf.nn.elu(x) + 1

        q_phi = phi(q)  # (B, Lq, D)
        k_phi = phi(k)  # (B, Lk, D)

        # 선형 어텐션 핵심: (Qφ * (Kφᵀ * V)) / (Qφ * (Kφᵀ * 1))
        kv = tf.matmul(k_phi, v, transpose_a=True)   # (B, D, D)
        z = 1 / (tf.matmul(q_phi, tf.reduce_sum(k_phi, axis=1, keepdims=True), transpose_b=True) + 1e-6)  # (B, Lq, 1)

        output = tf.matmul(q_phi, kv)  # (B, Lq, D)
        output = self.multiply([output, z])  # 스케일링

        output = self.o(output) + inputs  # 잔차 연결

        return output
    

class LearnableGlobalPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, dim)
        self.attention_weights = self.add_weight(
            shape=(input_shape[-1], 1),  # 각 토큰 feature 차원마다 1개 가중치
            initializer='glorot_uniform',
            trainable=True,
            name='attention_weights'
        )
        self.multiply = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch_size, seq_len, dim)
        # attention_scores: (batch_size, seq_len, 1)
        attention_scores = tf.matmul(inputs, self.attention_weights)  
        attention_scores = tf.nn.tanh(attention_scores)  # 비선형 활성화 (선택사항)
        attention_scores = tf.nn.softmax(attention_scores, axis=1)  # seq_len 축 기준 확률 분포
        
        x = self.multiply([inputs, attention_scores])  # (batch_size, seq_len, dim) * (batch_size, seq_len, 1)
        # 가중합: (batch_size, seq_len, dim) * (batch_size, seq_len, 1) -> (batch_size, dim)
        weighted_sum = tf.reduce_sum(x, axis=1)
        return weighted_sum

class LearnableRepeatVector(layers.Layer):
    def __init__(self, n, dim, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.dim = dim

    def build(self, input_shape):
        # self.weight shape를 (1, n, dim) 으로 만들어서 브로드캐스트 가능하게 변경
        self.weight = self.add_weight(
            shape=(1, self.n, self.dim),
            initializer='ones',
            trainable=True,
            name='repeat_weight'
        )
        self.multiply = layers.Multiply()
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, dim)
        x = tf.expand_dims(inputs, axis=1)  # (batch, 1, dim)
        x = tf.tile(x, [1, self.n, 1])      # (batch, n, dim)
        return self.multiply([x, self.weight])  # 브로드캐스트 문제 없이 곱해짐


def build_seprod_model(d_model):
    # 인코더 입력 및 임베딩 + 위치 임베딩
    encoder_input = Input(shape=(max_enc_len,), name='encoder_input')
    x_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(encoder_input)
    x = LearnablePositionalEmbedding(max_enc_len, d_model)(x_emb)
    x = Core(d_model)(x)
    x = LinearFWLayer(d_model)(x, x)
    x = Core(d_model)(x)

    # 컨텍스트 벡터 (인코더 출력 요약)
    context_vector = LearnableGlobalPooling()(x)  # 객체 생성 후 호출
    context_vector = LearnableRepeatVector(max_dec_len, d_model)(context_vector)  # 반복 및 학습 가중치

    # 디코더 입력 및 임베딩 + 위치 임베딩
    decoder_input = Input(shape=(max_dec_len,), name='decoder_input')
    y_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(decoder_input)
    y_pos = LearnablePositionalEmbedding(max_dec_len, d_model)(y_emb)
    y = Core(d_model)(y_pos)
    y = LinearFWLayer(d_model)(y, context_vector)
    y = Core(d_model)(y)

    # 출력 로짓
    logits = layers.Dense(vocab_size, dtype='float32')(y)

    return Model(inputs=[encoder_input, decoder_input], outputs=logits, name='SeProd')



d_model = 256
model = build_seprod_model(d_model)

import tensorflow as tf

steps_per_epoch = len(train_sentences) // batch_size
epochs = 1

initial_learning_rate = 1e-3
decay_steps = steps_per_epoch * epochs
alpha = 1e-6         # 최소 lr, 0에 가까운 값

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', Perplexity()]
)

model.summary()

model.fit(
    dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,

)

model.save("SeProD.h5")




def generate(model, sp, input_text, max_dec_len=128, temperature=0.7, verbose=False):
    """
    사용자 입력 문장을 받아 AI 응답 생성
    
    :param model: 훈련된 DustTransformer 모델
    :param sp: SentencePiece Tokenizer
    :param input_text: 사용자 입력 (str)
    :param max_dec_len: 최대 생성 길이
    :param temperature: 샘플링 온도 (낮을수록 greedy, 높을수록 창의적)
    :param verbose: 디버깅 메시지 출력 여부
    :return: 생성된 텍스트
    """
    start_id = sp.piece_to_id("<start>")
    end_id = sp.piece_to_id("<end>")
    
    # 인코더 입력 전처리
    enc_ids = sp.encode(input_text)
    enc_ids = enc_ids[:max_enc_len]
    enc_ids += [sp.pad_id()] * (max_enc_len - len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)

    if verbose:
        print("Encoder Input:", input_text)
        print("Encoded:", enc_ids)

    # 초기 디코더 입력 설정
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    generated_ids = []

    for step in range(max_dec_len):
        # 디코더 입력을 max_dec_len으로 패딩
        padded_dec_input = tf.pad(dec_input, [[0, 0], [0, max_dec_len - tf.shape(dec_input)[1]]],
                                  constant_values=sp.pad_id())

        # 전체 모델 예측
        decoder_output = model.predict([enc_tensor, padded_dec_input])
        
        # 현재 스텝의 로짓 추출
        logits = decoder_output[:, step, :]  # 현재 step 위치의 로짓

        if temperature == 0.:
            pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            logits = logits / temperature
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1)  # (1, )

        # 종료 토큰 체크
        if int(pred_id[0]) == end_id:
            break

        generated_ids.append(int(pred_id[0]))
        dec_input = tf.concat([dec_input, pred_id[:, tf.newaxis]], axis=1)

        if verbose:
            token_str = sp.decode([int(pred_id[0])])
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{token_str}'")

    decoded_text = sp.decode(generated_ids)
    return decoded_text

input_text = "회의록을 요약해 주세요."
response = generate(model, sp, input_text, temperature=0.7, verbose=True)
print("AI Response:", response)
