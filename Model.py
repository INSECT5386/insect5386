
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

class DynamicKernelGenerator(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, filters=64, hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.filters = filters
        self.hidden_dim = hidden_dim

        # 커널 생성 네트워크
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(kernel_size * filters, activation=None),
        ])

    def call(self, inputs):
        B, T, D = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        # 커널 생성: [B, T, K * F]
        kernels = self.net(inputs)

        # reshape to [B, T, K, F] → 각 타임스텝마다 다른 커널
        kernels = tf.reshape(kernels, [B, T, self.kernel_size, self.filters])

        return kernels

class DynamicConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def call(self, x, kernels):
        """
        x: [B, T, D]
        kernels: [B, T, K, F]
        returns: [B, T, F]
        """
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        K, F = self.kernel_size, self.filters

        # 패딩 추가 ( causal 또는 same )
        x = tf.pad(x, [[0, 0], [K // 2, K // 2], [0, 0]])

        # sliding window로 local patch 만들기
        patches = tf.image.extract_patches(
            images=tf.expand_dims(x, axis=1),  # [B, 1, T+K-1, D]
            sizes=[1, 1, K, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )  # [B, 1, T, K*D]
        patches = tf.reshape(patches, [B, T, K, D])  # [B, T, K, D]

        # 커널과 내적 계산
        output = tf.einsum('btkd,btkf->btf', patches, kernels)  # [B, T, F]

        return output

class DynamicConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_gen = DynamicKernelGenerator(kernel_size=kernel_size, filters=filters)
        self.conv = DynamicConv1D(filters, kernel_size)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        B, T, D = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        kernels = self.kernel_gen(x)  # [B, T, K, F]
        conv_out = self.conv(x, kernels)  # [B, T, F]

        out = self.norm(conv_out + x)  # residual connection
        out = tf.nn.relu(out)
        out = self.dropout(out, training=training)

        return out

def build_encoder(vocab_size, embed_dim=256, num_layers=4, filters=256):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)

    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)

    for _ in range(num_layers):
        x = DynamicConvBlock(filters)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name="encoder")

def build_decoder(vocab_size, embed_dim=256, num_layers=4, filters=256):
    dec_inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
    enc_outputs = tf.keras.Input(shape=(None, filters))

    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(dec_inputs)

    for _ in range(num_layers):
        x = DynamicConvBlock(filters)(x)

    # Attention (선택사항)
    x = tf.keras.layers.Attention()([x, enc_outputs])

    logits = tf.keras.layers.Dense(vocab_size)(x)

    return tf.keras.Model(inputs=[dec_inputs, enc_outputs], outputs=logits, name="decoder")

class DynamicConvS2S(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=256, filters=256, num_layers=4, **kwargs):
        super().__init__(**kwargs)
        self.encoder = build_encoder(vocab_size, embed_dim, num_layers, filters)
        self.decoder = build_decoder(vocab_size, embed_dim, num_layers, filters)

    def call(self, inputs, training=False):
        src, tgt = inputs
        memory = self.encoder(src)
        logits = self.decoder([tgt, memory])
        return logits


# 올바른 방식: DynamicConvS2S 모델 인스턴스 생성
model = DynamicConvS2S(vocab_size=vocab_size)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.summary()

# 학습 실행
model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)

# 저장
model.save("dynamic_conv_seq2seq_model.keras")

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

