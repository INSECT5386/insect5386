
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
train_sentences = train_sentences[:1280] # 예제용 소량
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
    input_text = sentence[:sep_index].strip()  # 질문 부분
    target_text = sentence[sep_index + len("<sep>"):].strip()  # 답변 부분

    # 인코더 입력: 질문 + <sep>
    enc_ids = sp.encode(input_text + " <sep>")[:max_enc_len - 1]  # <sep> 포함

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

# 넘파이 배열로 변환
encoder_inputs = np.array(encoder_inputs, dtype=np.int32)
decoder_inputs = np.array(decoder_inputs, dtype=np.int32)
targets = np.array(targets, dtype=np.int32)

def tf_dataset():
    def gen():
        for enc, dec, tgt in zip(encoder_inputs, decoder_inputs, targets):
            yield (enc, dec), tgt

    return tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), targets))

dataset = tf.data.Dataset.from_tensor_slices(((encoder_inputs, decoder_inputs), targets))
dataset = dataset.shuffle(buffer_size=1024)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("✅ 데이터셋 준비 완료")

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.activations import gelu, sigmoid, swish

# 사용할 하이퍼파라미터
vocab_size = 32000
embed_dim = 256
max_enc_len = 128
max_dec_len = 128
pad_id = sp.piece_to_id("<pad>")

# ---- Encoder ----
def build_encoder():
    inputs = Input(shape=(None,), dtype=tf.int32)
    x = Embedding(vocab_size, embed_dim)(inputs)
    
    # GeLU × Sigmoid
    g = tf.keras.activations.gelu(x)
    s = tf.keras.activations.sigmoid(x)
    x = g * s  # Element-wise multiplication
    
    return Model(inputs=inputs, outputs=x, name="SimpleEncoder")

# ---- Decoder ----
def build_decoder():
    dec_inputs = Input(shape=(None,), dtype=tf.int32)
    enc_outputs = Input(shape=(None, embed_dim), dtype=tf.float32)

    x = Embedding(vocab_size, embed_dim)(dec_inputs)
    
    # SiLU (Swish)
    x = tf.keras.activations.swish(x)
    
    # Concat with encoder output
    batch_size = tf.shape(dec_inputs)[0]
    seq_len = tf.shape(dec_inputs)[1]
    
    # Repeat encoder output to match decoder's sequence length
    enc_expanded = tf.expand_dims(enc_outputs, axis=1)  # [B, 1, T_enc, D]
    enc_tiled = tf.tile(enc_expanded, [1, seq_len, 1, 1])  # [B, T_dec, T_enc, D]
    enc_flattened = tf.reshape(enc_tiled, [batch_size, seq_len, -1])  # [B, T_dec, D*T_enc]

    x = Concatenate(axis=-1)([x, enc_flattened])
    
    # SiLU × Sigmoid
    x = Dense(embed_dim)(x)
    g = tf.keras.activations.swish(x)
    s = tf.keras.activations.sigmoid(x)
    x = g * s
    
    logits = Dense(vocab_size)(x)
    
    return Model(inputs=[dec_inputs, enc_outputs], outputs=logits, name="SimpleDecoder")


class SimpleS2S(tf.keras.Model):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder()
        self.decoder = build_decoder()

    def call(self, inputs):
        src, tgt = inputs
        memory = self.encoder(src)
        logits = self.decoder([tgt, memory])
        return logits

model = SimpleS2S(vocab_size=vocab_size)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 데이터셋은 기존 dataset 그대로 사용
model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)

model.save("simple_s2s_model.keras")

def generate_response(model, sp, input_text, max_len=128):
    start_id = sp.piece_to_id("<start>")
    sep_id = sp.piece_to_id("<sep>")
    end_id = sp.piece_to_id("<end>")

    # 질문 처리
    enc_tokens = sp.encode(input_text + " <sep>")
    enc_padded = enc_tokens + [pad_id] * (max_enc_len - len(enc_tokens))
    enc_input = np.array([enc_padded], dtype=np.int32)

    # 디코더 초기 입력
    dec_input = np.array([[start_id]], dtype=np.int32)

    generated = []

    for _ in range(max_len):
        logits = model.predict((enc_input, dec_input))
        next_token = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)

        if int(next_token[0]) == end_id:
            break

        generated.append(int(next_token[0]))
        dec_input = tf.concat([dec_input, tf.expand_dims(next_token, axis=0)], axis=-1)

    return sp.decode(generated)

input_text = "안녕하세요"

generate_response(model, sp, input_text)
