import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
import numpy as np
import sentencepiece as spm
import pandas as pd
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
train_sentences = train_sentences[:10]  # 예제용 소량
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
max_enc_len = 128   # 인코더 최대 길이 (질문 부분)
max_dec_len = 128   # 디코더 최대 길이 (답변 부분)
batch_size = 32

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
    enc_ids = sp.encode(input_text + " <sep>")[:max_enc_len]

    # 디코더 입력: <start> + 답변
    dec_input_ids = [start_id] + sp.encode(target_text)[:max_dec_len - 1]

    # 정답 라벨: 답변 + <end>
    target_ids = sp.encode(target_text + " <end>")[:max_dec_len]

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
    tf.int32             # 타겟에 대한 타입
)

output_shapes = (
    (tf.TensorShape([max_enc_len]), tf.TensorShape([max_dec_len])), # 두 개의 입력 텐서에 대한 모양
    tf.TensorShape([max_dec_len])                                   # 타겟에 대한 모양
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_types=output_types,
    output_shapes=output_shapes
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("dataset ok")

import tensorflow as tf

class RecurrentFFN(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim * 4
        self.state_size = self.hidden_dim

        # 각 레이어를 개별적으로 선언
        self.update_gate_dense = tf.keras.layers.Dense(self.hidden_dim)
        self.update_gate_act = tf.keras.layers.Activation('sigmoid')

        self.reset_gate_dense = tf.keras.layers.Dense(self.hidden_dim)
        self.reset_gate_act = tf.keras.layers.Activation('sigmoid')

        self.gate_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)
        self.up_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)

        self.down_proj = tf.keras.layers.Dense(input_dim, use_bias=True)

        self.norm_hidden = tf.keras.layers.LayerNormalization()
        self.norm_output = tf.keras.layers.LayerNormalization()

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        # 입력 크기 기반으로 내부 가중치를 명시적으로 생성
        # input_shape는 (batch_size, sequence_length, features) 형태일 수 있으므로 마지막 차원을 사용
        # self.input_dim은 input_shape의 마지막 차원이어야 함.
        # RecurrentFFN이 RNN 셀처럼 작동한다면, input_shape는 (batch_size, features)
        # 만약 (batch_size, seq_len, features)로 들어온다면 RNN Layer에서 각 타임스텝의 input_shape이 (batch_size, features)가 됨.
        # 따라서 build에서 input_shape[-1]을 사용하는 것은 맞음.
        
        # 다만, hidden_dim은 input_dim * 4로 설정될 수 있으므로,
        # build 메서드 호출 시 전달되는 input_shape는 x의 shape만 해당될 수 있음.
        # hidden_state의 shape도 고려해야 combined의 input_shape가 올바르게 계산됨.
        # 여기서는 RecurrentFFN이 RNN의 Cell로 사용된다고 가정하고 build의 input_shape가 x의 shape라고 가정.

        # Keras Layer의 build 메서드는 일반적으로 첫 번째 인자로 (batch_size, ...) 형태의 input_shape를 받음
        # RecurrentFFN의 build는 x의 input_shape를 받음.
        # 따라서 combined의 shape는 (None, input_shape[-1] + self.hidden_dim)이 됨.
        combined_input_dim = input_shape[-1] + self.hidden_dim

        self.update_gate_dense.build((None, combined_input_dim))
        self.reset_gate_dense.build((None, combined_input_dim))
        self.gate_proj.build((None, combined_input_dim))
        self.up_proj.build((None, combined_input_dim))
        self.down_proj.build((None, self.hidden_dim))
        self.built = True

    def call(self, x, hidden_state, training=False):
        # hidden_state가 튜플일 경우 첫 번째 요소를 추출
        # Keras RNN 래퍼가 셀의 상태를 튜플로 묶어서 전달할 수 있기 때문
        if isinstance(hidden_state, (list, tuple)):
            current_hidden_state = hidden_state[0]
        else:
            current_hidden_state = hidden_state

        # 이제 x와 current_hidden_state는 모두 (batch_size, feature_dim) 형태의 2차원 텐서여야 합니다.
        combined = tf.concat([x, current_hidden_state], axis=-1)

        update_gate = self.update_gate_act(self.update_gate_dense(combined))
        reset_gate = self.reset_gate_act(self.reset_gate_dense(combined))

        gated_hidden = reset_gate * current_hidden_state # 여기도 current_hidden_state 사용
        candidate_combined = tf.concat([x, gated_hidden], axis=-1)

        gate = self.gate_proj(candidate_combined)
        up = self.up_proj(candidate_combined)
        swiglu_output = tf.nn.silu(gate) * up

        new_hidden_state = (1 - update_gate) * current_hidden_state + update_gate * swiglu_output
        new_hidden_state = self.norm_hidden(new_hidden_state)

        output = self.down_proj(new_hidden_state)
        output = self.norm_output(output)
        output = self.dropout(output, training=training)

        # RNN 셀로 사용될 때는 (출력, 다음_상태) 튜플을 반환해야 함.
        # Keras의 RNN 레이어가 이 형태를 기대함.
        return output, new_hidden_state

    @property
    def state_size(self):
        # RNN Cell의 state_size 속성은 Keras RNN Layer가 초기 상태를 생성하고 관리할 때 사용합니다.
        # 이는 일반적으로 단일 정수 또는 상태 텐서들의 모양을 나타내는 튜플입니다.
        # 여기서는 단일 hidden_state를 사용하므로 hidden_dim을 반환합니다.
        return self.hidden_dim

    def get_initial_state(self, batch_size=None, dtype=None):
        # dtype이 None일 경우 self.dtype (레이어의 기본 dtype)을 사용하거나
        # 아니면 tf.float32와 같은 명시적인 타입을 사용합니다.
        # Keras 내부에서 RNN Cell의 dtype을 기반으로 이 메서드를 호출할 때
        # dtype 인자를 전달하므로, 해당 dtype을 우선적으로 사용하는 것이 좋습니다.
        actual_dtype = dtype if dtype is not None else self.dtype if hasattr(self, 'dtype') else tf.float32
        return tf.zeros(shape=[batch_size, self.state_size], dtype=actual_dtype)



# 인코더
encoder_input = tf.keras.Input(shape=(max_enc_len,))
encoder_emb = tf.keras.layers.Embedding(vocab_size, 200)(encoder_input)

rnn_cell = RecurrentFFN(input_dim=200, hidden_dim=200)
encoder = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True, name='encoder')
encoder_output, encoder_final_state = encoder(encoder_emb)

# 디코더
decoder_input = tf.keras.Input(shape=(max_dec_len,))
decoder_emb = tf.keras.layers.Embedding(vocab_size, 200)(decoder_input)

rnn_cell_decoder = RecurrentFFN(input_dim=200, hidden_dim=200)

decoder = tf.keras.layers.RNN(
    rnn_cell_decoder,
    return_sequences=True,
    return_state=True,
    name='decoder',
)

decoder_output, _ = decoder(decoder_emb, initial_state=encoder_final_state)

# 출력층
decoder_dense = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(vocab_size)
)
decoder_outputs = decoder_dense(decoder_output)

# 모델 정의
model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.summary()
model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)

def generate(model, sp, input_text, max_dec_len=128, temperature=0.7, verbose=False):
    """
    사용자 입력 문장을 받아 AI 응답 생성
    :param model: 훈련된 Seq2Seq 모델
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

    # 인코더 실행 (인코더 임베딩 -> RNN)
    encoder_emb_layer = model.get_layer('embedding')  # 인코더 임베딩
    encoder_rnn_layer = model.get_layer('encoder')    # 인코더 RNN

    encoder_emb_out = encoder_emb_layer(enc_tensor)
    encoder_output, encoder_state = encoder_rnn_layer(encoder_emb_out, training=False)

    # 디코더 준비
    decoder_emb_layer = model.get_layer('embedding_1')  # 디코더 임베딩
    decoder_rnn_layer = model.get_layer('decoder')      # 디코더 RNN
    decoder_dense_layer = model.get_layer('time_distributed')  # TimeDistributed(Dense)

    # 디코더 초기 입력: <start>
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    current_state = encoder_state
    generated_ids = []

    for step in range(max_dec_len):
        dec_emb = decoder_emb_layer(dec_input)  # 입력 임베딩

        decoder_output, next_state = decoder_rnn_layer(
            dec_emb, initial_state=current_state, training=False
        )

        logits = decoder_dense_layer(decoder_output)  # (batch_size, 1, vocab_size)
        logits = tf.squeeze(logits, axis=1)  # (batch_size, vocab_size)

        # 온도 조절 샘플링
        if temperature == 0.:
            pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            logits = logits / temperature
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1)  # (1,)

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

input_text = "안녕하세요"

generate(model, sp, input_text)
