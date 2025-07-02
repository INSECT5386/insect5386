
import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
from tensorflow.keras.initializers import RandomNormal
import sentencepiece as spm  
import requests
from tensorflow.keras.initializers import RandomNormal

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import RandomNormal


import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dropout

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
train_sentences = train_sentences[:200000] # 예제용 소량
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


class GLALayer(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.gate_proj = layers.Dense(dim)
        self.sigmoid = tf.keras.activations.sigmoid
        self.Wo = layers.Dense(dim)

    def call(self, inputs, context):
        """
        inputs: 디코더 입력 (B, Tq, dim)
        context: 인코더 문맥 (B, Tk, dim)
        """
        # gate signal 생성
        gate_signal = self.gate_proj(context)  # (B, Tk, dim)

        # QK 유사도 대신 단순 내적
        attn_weights = tf.einsum('bkd,bqd->bkq', gate_signal, inputs)
        # => (B, Tk, Tq), softmax 후에는 (B, Tk, Tq): 각 문맥 토큰이 입력에 미치는 영향

        # Softmax 적용
        attn_weights = tf.nn.softmax(attn_weights, axis=1)  # Tk 방향으로 softmax

        # 가중합
        output = tf.einsum('bkq,bkd->bqd', attn_weights, context)
        # => (B, Tq, dim)

        # Output projection
        return self.Wo(output)
       

class SpatialGatingUnit(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.norm = layers.LayerNormalization()
        self.gate = layers.Dense(dim)
        self.mul = layers.Multiply()

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, dim)
        x = self.norm(inputs)
        gate_values = self.gate(x)  # (batch_size, seq_len, dim)
        return self.mul([inputs, gate_values])  # element-wise multiplication

class GMLPBlock(layers.Layer):
    def __init__(self, dim, expansion_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.norm = layers.LayerNormalization()
        self.up_proj = layers.Dense(dim * expansion_factor)
        self.act = layers.Activation('gelu')
        self.sgu = SpatialGatingUnit(dim)
        self.down_proj = layers.Dense(dim)
        self.add = layers.Add()

    def call(self, inputs):
        x = self.norm(inputs)
        x = self.up_proj(x)
        x = self.act(x)
        x = self.sgu(x)
        x = self.down_proj(x)
        return self.add([x, inputs])  # Residual connection

class MatmulBlock(layers.Layer):
    def __init__(self, dim, expansion_factor=1, **kwargs):
        super().__init__(**kwargs)
        self.matmul = tf.matmul

    def call(self, x, emb):
        output = self.matmul(x, emb.embedding, transpose_b=True)
        return output
                             
d_model = 256
dropout_rate = 0.1
# ===== 모델 구성 =====
# 인코더 경로
encoder_input = Input(shape=(max_enc_len,), name='encoder_input')
x_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(encoder_input)
# 인코더 경로
x = LearnablePositionalEmbedding(max_enc_len, d_model)(x_emb)
for _ in range(4):  # 인코더 블록 반복
    x = GMLPBlock(d_model)(x)
context_vector = x

decoder_input = Input(shape=(max_dec_len,), name='decoder_input')
y_emb = layers.Embedding(input_dim=vocab_size, output_dim=d_model)(decoder_input)
# 디코더 경로
y = LearnablePositionalEmbedding(max_dec_len, d_model)(y_emb)
for _ in range(4):  # 디코더 블록 반복
    y = GMLPBlock(d_model)(y)
    y = GLALayer(d_model)(y, context_vector)

output = layers.Dense(128)(y)
output = layers.Activation(tf.nn.gelu)(output)
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
