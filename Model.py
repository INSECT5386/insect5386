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
train_sentences = train_sentences[:300]  # 예제용 소량
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
        yield (
            {'encoder_input': enc, 'decoder_input': dec},
            tgt
        )

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
print("dataset ok")

class RecurrentFFN(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim * 4

        # Update Gate and Reset Gate (like GRU)
        self.update_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.reset_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.Activation('sigmoid')
        ])

        # Candidate hidden state (SwiGLU 기반)
        self.gate_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)
        self.up_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)

        # Down projection
        self.down_proj = tf.keras.layers.Dense(input_dim, use_bias=True)

        # Layer Normalizations
        self.norm_hidden = tf.keras.layers.LayerNormalization()
        self.norm_output = tf.keras.layers.LayerNormalization()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def build(self, input_shape):
        pass

    def call(self, x, hidden_state, training=False):
        # Update Gate & Reset Gate 계산
        combined = tf.concat([x, hidden_state], axis=-1)
        update_gate = self.update_gate(combined)
        reset_gate = self.reset_gate(combined)

        # Reset 적용 후 candidate 생성
        gated_hidden = reset_gate * hidden_state
        candidate_combined = tf.concat([x, gated_hidden], axis=-1)

        # SwiGLU 활성화
        gate = self.gate_proj(candidate_combined)
        up = self.up_proj(candidate_combined)
        swiglu_output = tf.nn.silu(gate) * up

        # Hidden State 업데이트
        new_hidden_state = (1 - update_gate) * hidden_state + update_gate * swiglu_output
        new_hidden_state = self.norm_hidden(new_hidden_state)

        # Final output
        output = self.down_proj(new_hidden_state)
        output = self.norm_output(output)
        output = self.dropout(output, training=training)

        return output, new_hidden_state

    def get_initial_state(self, batch_size=None, dtype=None):
        """Returns initial hidden state"""
        return tf.zeros(shape=[batch_size, self.hidden_dim], dtype=dtype)


# 가정된 파라미터
vocab_size = 10000
max_len = 256

# 인코더
encoder_input = tf.keras.Input(shape=(max_len,))
encoder_emb = tf.keras.layers.Embedding(vocab_size, 50)(encoder_input)

rnn_cell = RecurrentFFN(input_dim=50, hidden_dim=200)
encoder = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True, name='encoder')
encoder_output, encoder_final_state = encoder(encoder_emb)

# 디코더
decoder_input = tf.keras.Input(shape=(max_len,))
decoder_emb = tf.keras.layers.Embedding(vocab_size, 50)(decoder_input)

rnn_cell_decoder = RecurrentFFN(input_dim=50, hidden_dim=200)

decoder = tf.keras.layers.RNN(
    rnn_cell_decoder,
    return_sequences=True,
    return_state=True,
    name='decoder',
)

decoder_output, _ = decoder(decoder_emb, initial_state=encoder_final_state)

# 출력층
decoder_dense = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(vocab_size, activation='softmax')
)
decoder_outputs = decoder_dense(decoder_output)

# 모델 정의
model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
