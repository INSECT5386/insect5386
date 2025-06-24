
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
    input_text = sentence[:sep_index].strip() # 질문 부분
    target_text = sentence[sep_index + len("<sep>"):].strip() # 답변 부분

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

dataset = dataset.shuffle(16384).batch(batch_size).prefetch(tf.data.AUTOTUNE)
print("dataset ok")

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

# 모델 정의
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.summary()
model.fit(dataset, epochs=1, steps_per_epoch=len(train_sentences) // batch_size)
model.save("model.keras") # 최선의 선택
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
    encoder_emb_layer = model.get_layer('embedding') # 인코더 임베딩
    encoder_rnn_layer = model.get_layer('encoder') # 인코더 RNN

    encoder_emb_out = encoder_emb_layer(enc_tensor)
    encoder_output, encoder_state = encoder_rnn_layer(encoder_emb_out, training=False)

    # 디코더 준비
    decoder_emb_layer = model.get_layer('embedding_1') # 디코더 임베딩
    decoder_rnn_layer = model.get_layer('decoder') # 디코더 RNN
    decoder_dense_layer = model.get_layer('time_distributed') # TimeDistributed(Dense)

    # 디코더 초기 입력: <start>
    dec_input = tf.constant([[start_id]], dtype=tf.int32)
    current_state = encoder_state
    generated_ids = []

    for step in range(max_dec_len):
        dec_emb = decoder_emb_layer(dec_input) # 입력 임베딩

        decoder_output, next_state = decoder_rnn_layer(
            dec_emb, initial_state=current_state, training=False
        )

        logits = decoder_dense_layer(decoder_output) # (batch_size, 1, vocab_size)
        logits = tf.squeeze(logits, axis=1) # (batch_size, vocab_size)

        # 온도 조절 샘플링
        if temperature == 0.:
            pred_id = tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            logits = logits / temperature
            pred_id = tf.random.categorical(logits, 1, dtype=tf.int32)

        pred_id = tf.squeeze(pred_id, axis=1) # (1,)

        # 종료 토큰 체크
        if int(pred_id[0]) == end_id:
            break

        generated_ids.append(int(pred_id[0]))
        dec_input = pred_id[:, tf.newaxis] # 다음 입력으로 업데이트
        current_state = next_state

        if verbose:
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{sp.decode([int(pred_id[0])])}'")

    decoded_text = sp.decode(generated_ids)
    return decoded_text

input_text = "안녕하세요"

generate(model, sp, input_text)
