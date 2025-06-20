import json  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
import sentencepiece as spm  
import requests
from tensorflow.keras.layers import Embedding, RNN, Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np


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
max_enc_len = 128   # 인코더 최대 길이 (질문 부분)
max_dec_len = 128   # 디코더 최대 길이 (답변 부분)
batch_size = 64

# ⬇️ 전처리 결과 저장할 리스트
encoder_inputs = []
decoder_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index].strip()      # 질문 부분
    target_text = sentence[sep_index + len("<sep>"):].strip()  # 답변 부분

    # 인코더 입력: 질문 + <sep>
    enc_ids = text_to_ids(input_text + " <sep>")[:max_enc_len]

    # 디코더 입력: <start> + 답변
    dec_input_ids = [start_id] + text_to_ids(target_text)[:max_dec_len - 1]

    # 정답 라벨: 답변 + <end>
    target_ids = text_to_ids(target_text + " <end>")[:max_dec_len]

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

import tensorflow as tf

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for enc, dec, tgt in zip(encoder_inputs, decoder_inputs, targets):
        yield (
            {'encoder_input': enc, 'decoder_input': dec},
            tgt
        )

# output_types 및 output_shapes 정확히 지정
output_types = (
    {
        'encoder_input': tf.int32,
        'decoder_input': tf.int32
    },
    tf.int32
)

output_shapes = (
    {
        'encoder_input': tf.TensorShape([max_enc_len]),
        'decoder_input': tf.TensorShape([max_dec_len])
    },
    tf.TensorShape([max_dec_len])
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=output_types,
    output_shapes=output_shapes
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


class VecAwCell(Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(VecAwCell, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.state_size = [tf.TensorShape(units), tf.TensorShape(units)]  # [shortterm, longterm]

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # FRU 가중치
        self.kernel_x = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='he_normal',
            name='kernel_x'
        )
        self.kernel_h = self.add_weight(
            shape=(self.units, self.units),
            initializer='he_normal',
            name='kernel_h'
        )

        # 멀티헤드 어텐션 파라미터
        self.num_heads = 8
        self.head_dim = self.units // self.num_heads

        self.W_q = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_q'
        )
        self.W_k = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_k'
        )
        self.W_v = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_v'
        )
        self.W_o = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_o'
        )

        # 게이트 메커니즘
        self.forget_gate = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='forget_gate'
        )
        self.input_gate = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='input_gate'
        )

        # 정규화 & 드롭아웃
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        else:
            dtype = tf.float32

        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype)
        ]

    def multi_head_attention(self, query, key, value):
        """멀티헤드 어텐션 구현"""
        batch_size = tf.shape(query)[0]

        q = tf.matmul(query, self.W_q)
        k = tf.matmul(key, self.W_k)
        v = tf.matmul(value, self.W_v)

        # 분할: (batch_size, num_heads, head_dim)
        q = tf.reshape(q, (batch_size, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, self.num_heads, self.head_dim))

        # 스코어 계산
        dk = tf.cast(self.head_dim, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scores, axis=-1)

        # 가중합
        attended = tf.matmul(weights, v)

        # 다시 합침
        attended = tf.reshape(attended, (batch_size, self.units))

        # 최종 선형 변환
        output = tf.matmul(attended, self.W_o)

        return output

    def call(self, inputs, states, training=None):
        h_prev, longterm_prev = states

        x_t_proj = tf.matmul(inputs, self.kernel_x)
        h_prev_proj = tf.matmul(h_prev, self.kernel_h)
        x = x_t_proj + h_prev_proj

        # 게이트 계산
        forget_gate = tf.sigmoid(tf.matmul(x, self.forget_gate))
        input_gate = tf.sigmoid(tf.matmul(x, self.input_gate))

        # LSTM 스타일의 셀 상태 업데이트
        candidate = tf.tanh(x)
        longterm = forget_gate * longterm_prev + input_gate * candidate

        mediumterm = tf.nn.swish(x)
        shortterm = tf.nn.swish(longterm + mediumterm)

        # 정규화
        shortterm = self.layer_norm1(shortterm)
        longterm = self.layer_norm2(longterm)

        # 멀티헤드 어텐션 적용 (Self-Attention)
        attended = self.multi_head_attention(shortterm, shortterm, shortterm)

        # 잔차 연결
        output = shortterm + attended

        # 드롭아웃
        if training:
            output = self.dropout(output, training=training)

        return output, [output, longterm]


# 공유 임베딩 레이어
shared_embedding = Embedding(
    input_dim=48000,
    output_dim=256,
    mask_zero=True,
    name='shared_embedding'
)
# 🔥 명시적으로 build() 호출
shared_embedding.build(input_shape=(None,))  # (batch_size, seq_len)


class VecAwEncoder(Model):
    def __init__(self, shared_embedding, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = shared_embedding
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

    def call(self, inputs, training=None):
        mask = self.embedding.compute_mask(inputs)
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        outputs, h_state, longterm_state = self.rnn(x, mask=mask, training=training)
        return outputs, [h_state, longterm_state]


class VecAwDecoder(Model):
    def __init__(self, shared_embedding, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = shared_embedding
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

        # 출력층 - 임베딩과 가중치 공유
        vocab_size = int(shared_embedding.input_dim)
        self.output_layer = Dense(vocab_size, use_bias=False, name='output_layer')
        self.output_layer.build((None, hidden_units))
        self.output_layer.set_weights([tf.transpose(shared_embedding.weights[0])])
    def call(self, inputs, initial_state, training=None):
        mask = self.embedding.compute_mask(inputs)
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)

        rnn_out, h_state, longterm_state = self.rnn(x, initial_state=initial_state, mask=mask, training=training)
        logits = self.output_layer(rnn_out)

        return logits, [h_state, longterm_state]


class VecAwSeq2Seq(Model):
    def __init__(self, shared_embedding, hidden_units, 
                 start_id=sp.piece_to_id("<start>"), end_id=sp.piece_to_id("<end>"), max_length=128, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VecAwEncoder(shared_embedding, hidden_units, dropout_rate)
        self.decoder = VecAwDecoder(shared_embedding, hidden_units, dropout_rate)

        self.start_token_id = start_id
        self.end_token_id = end_id
        self.max_length = max_length

    def build(self, input_shape):
        # encoder와 decoder가 사용할 input shape 전달
        if isinstance(input_shape, dict):
            enc_shape = input_shape['encoder_input']
            dec_shape = input_shape['decoder_input']
        else:
            enc_shape, dec_shape = input_shape[0], input_shape[1]
        
        self.encoder.build(enc_shape)
        self.decoder.build(dec_shape)
        self.built = True

    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            encoder_input, decoder_input = inputs
        else:
            encoder_input = inputs['encoder_input']
            decoder_input = inputs['decoder_input']

        encoder_output, encoder_state = self.encoder(encoder_input, training=training)
        decoder_output, _ = self.decoder(decoder_input, initial_state=encoder_state, training=training)

        return decoder_output

    def get_config(self):
        config = super().get_config()
        config.update({
            'start_token_id': self.start_token_id,
            'end_token_id': self.end_token_id,
            'max_length': self.max_length
        })
        return config

model = VecAwSeq2Seq(shared_embedding, hidden_units=256)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
# 예제 입력으로 build 유도
test_encoder_input = tf.constant([ [1] * max_enc_len ])  # (1, 128)
test_decoder_input = tf.constant([ [1] * max_dec_len ])  # (1, 128)

_ = model((test_encoder_input, test_decoder_input))  # build 유도
model.fit(dataset, epochs=10, steps_per_epoch=len(train_sentences) // batch_size)
model.summary()

def generate(model, sp, input_text, max_dec_len=128, temperature=0.7, verbose=False):
    """
    사용자 입력 문장을 받아 AI 응답 생성
    :param model: 훈련된 VecAwSeq2Seq 모델
    :param sp: SentencePiece Tokenizer
    :param input_text: 사용자 입력 (str)
    :param max_dec_len: 최대 생성 길이
    :param temperature: 샘플링 온도 (낮을수록 greedy, 높을수록 창의적)
    :param verbose: 디버깅 메시지 출력 여부
    :return: 생성된 텍스트
    """
    start_id = sp.piece_to_id("<start>")
    end_id = sp.piece_to_id("<end>")
    sep_id = sp.piece_to_id("<sep>")

    # 인코더 입력 전처리
    enc_ids = sp.encode(input_text + " <sep>")
    enc_ids = enc_ids[:max_enc_len]
    enc_ids += [sp.pad_id()] * (max_enc_len - len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)

    if verbose:
        print("Encoder Input:", input_text)
        print("Encoded:", enc_ids)

    # 인코더 실행
    encoder_output, encoder_state = model.encoder(enc_tensor, training=False)

    # 디코더 초기 입력: <start>
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    current_state = encoder_state
    generated_ids = []

    for step in range(max_dec_len):
        decoder_output, next_state = model.decoder(
            dec_input, initial_state=current_state, training=False
        )

        logits = decoder_output[:, -1, :]  # 마지막 타임스텝의 로짓

        # 온도 조절 샘플링
        if temperature == 0.:
            pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            logits = logits / temperature
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1)

        # 종료 토큰 체크
        if int(pred_id[0]) == end_id:
            break

        generated_ids.append(int(pred_id[0]))
        dec_input = pred_id[:, tf.newaxis]  # 다음 입력으로 업데이트
        current_state = next_state

        if verbose:
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{sp.decode([int(pred_id[0])])}'")

    decoded_text = sp.decode(generated_ids)
    return decoded_text

# 예시 질문
user_input = "오늘 날씨가 어때?"

# AI 답변 생성
response = generate(model, sp, user_input, max_dec_len=64, temperature=0.7, verbose=True)
print("\nAI 답변:", response)
