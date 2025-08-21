import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, RNN, Attention
from tensorflow.keras.models import Model
import numpy as np
import json
import requests
import sentencepiece as spm

# =======================
# 파일 다운로드
# =======================
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

download_file(
    'https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/dataset.jsonl?download=true',
    'VeTrans.jsonl'
)
download_file(
    'https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/kolig_unigram.model?download=true',
    'ko_unigram.model'
)

# =======================
# 토크나이저 로드
# =======================
sp = spm.SentencePieceProcessor()
sp.load('ko_unigram.model')
vocab_size = sp.get_piece_size()

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>") if sp.piece_to_id("<start>") != -1 else (sp.bos_id() or 1)
end_id = sp.piece_to_id("<end>") if sp.piece_to_id("<end>") != -1 else (sp.eos_id() or 2)

print("TOKENS:", {"pad": pad_id, "start": start_id, "end": end_id, "vocab": vocab_size})

# =======================
# 데이터셋 생성
# =======================
max_len = 100
batch_size = 64

def data_generator(file_path, max_len=max_len, pad_id=pad_id, start_id=start_id, end_id=end_id, max_samples=None):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            item = json.loads(line)
            if "conversations" not in item or len(item["conversations"]) < 2:
                continue
            src = item["conversations"][0].get("value", "").strip()
            tgt = item["conversations"][1].get("value", "").strip()
            if len(src) < 2 or len(tgt) < 2 or src == tgt:
                continue
            def encode_text(text):
                ids = sp.encode(text, out_type=int)
                ids = [start_id] + ids + [end_id]
                if len(ids) < max_len:
                    ids += [pad_id]*(max_len - len(ids))
                else:
                    ids = ids[:max_len]
                    if ids[-1] != end_id:
                        ids[-1] = end_id
                return np.array(ids, dtype=np.int32)
            yield (encode_text(src), encode_text(tgt))
            count += 1

#3097345

stream_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("VeTrans.jsonl", max_samples=30),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)
stream_dataset = stream_dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 디코더 입력/타겟 매핑
def map_fn(src, tgt):
    dec_input = tgt[:, :-1]
    dec_target = tgt[:, 1:]
    return ({"enc_inputs": src, "dec_inputs": dec_input}, dec_target)

train_ds = stream_dataset.map(map_fn)
print("✅ train_ds 준비 완료:", train_ds)

# =======================
# RecurrentFFN
# =======================
class RecurrentFFN(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim*4
        self.state_size = self.hidden_dim

        self.update_gate_dense = Dense(self.hidden_dim)
        self.update_gate_act = tf.keras.activations.sigmoid

        self.reset_gate_dense = Dense(self.hidden_dim)
        self.reset_gate_act = tf.keras.activations.sigmoid

        self.gate_proj = Dense(self.hidden_dim)
        self.up_proj = Dense(self.hidden_dim)
        self.down_proj = Dense(input_dim)

        self.norm_hidden = LayerNormalization()
        self.norm_output = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

    def build(self, input_shape):
        combined_input_dim = input_shape[-1] + self.hidden_dim
        self.update_gate_dense.build((None, combined_input_dim))
        self.reset_gate_dense.build((None, combined_input_dim))
        self.gate_proj.build((None, combined_input_dim))
        self.up_proj.build((None, combined_input_dim))
        self.down_proj.build((None, self.hidden_dim))

        # LayerNormalization build
        self.norm_hidden.build((None, self.hidden_dim))
        self.norm_output.build((None, self.input_dim))

        self.built = True


    def call(self, x, hidden_state, training=False):
        if isinstance(hidden_state, (list, tuple)):
            h = hidden_state[0]
        else:
            h = hidden_state
        combined = tf.concat([x,h], axis=-1)
        u = self.update_gate_act(self.update_gate_dense(combined))
        r = self.reset_gate_act(self.reset_gate_dense(combined))
        gated_h = r*h
        cand = tf.concat([x,gated_h], axis=-1)
        gate = self.gate_proj(cand)
        up = self.up_proj(cand)
        swiglu = tf.nn.silu(gate)*up
        new_h = self.norm_hidden((1-u)*h + u*swiglu + h)  # residual 추가
        out = self.down_proj(new_h)
        out = self.norm_output(out)
        return out, new_h

    def get_initial_state(self, batch_size=None, dtype=None):
        return tf.zeros([batch_size, self.state_size], dtype=dtype or tf.float32)

# =======================
# 학습 모델 정의
# =======================
max_enc_len = max_len
max_dec_len = max_len

# 인코더
enc_input = Input(shape=(max_enc_len,), name="enc_inputs")
enc_emb = Embedding(vocab_size,128)(enc_input)
enc_rnn_cell = RecurrentFFN(128,128)
enc_rnn = RNN(enc_rnn_cell, return_sequences=True, return_state=True)
enc_seq, enc_state = enc_rnn(enc_emb)

# 디코더
dec_input = Input(shape=(max_dec_len-1,), name="dec_inputs")
dec_emb = Embedding(vocab_size,128)(dec_input)
dec_rnn_cell = RecurrentFFN(128,128)
dec_rnn = RNN(dec_rnn_cell, return_sequences=True, return_state=True)
dec_out, _ = dec_rnn(dec_emb, initial_state=enc_state)

attn_out = Attention()([dec_out, enc_seq])
attn_out = LayerNormalization()(attn_out)
attn_out = Dense(128, activation='relu')(attn_out)  # FFN 추가
logits = Dense(vocab_size)(attn_out)


train_model = Model([enc_input, dec_input], logits)
train_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
print(train_model.summary())
epochs = 1
train_model.fit(train_ds, epochs=epochs)
# =======================
# 추론 모델 정의
# =======================
# 인코더 모델
encoder_model = Model(enc_input, [enc_seq, enc_state])

# 디코더 모델
dec_token_input = Input(shape=(1,), name="dec_token_input")
dec_state_input = Input(shape=(128,), name="dec_state_input")
enc_seq_input = Input(shape=(max_enc_len,128), name="enc_seq_input")  # 인코더 시퀀스 feed

dec_emb_layer = Embedding(vocab_size,128)
dec_rnn_cell_infer = RecurrentFFN(128,128)
dec_rnn_layer = RNN(dec_rnn_cell_infer, return_sequences=True, return_state=True)
dec_dense_layer = Dense(vocab_size)

dec_embedded = dec_emb_layer(dec_token_input)
dec_out_step, dec_state_out = dec_rnn_layer(dec_embedded, initial_state=dec_state_input)
dec_attn_out = Attention()([dec_out_step, enc_seq_input])
dec_logits_step = dec_dense_layer(dec_attn_out)

decoder_model = Model(
    [dec_token_input, dec_state_input, enc_seq_input],
    [dec_logits_step, dec_state_out]
)
encoder_model.save("vechat_encoder.keras")
decoder_model.save("vechat_decoder.keras", include_optimizer=False)

# =======================
# 추론 함수
# =======================
def generate(input_text, max_dec_len=64, temperature=0.7, verbose=False):
    enc_ids = sp.encode(input_text)[:max_enc_len]
    enc_ids += [pad_id]*(max_enc_len-len(enc_ids))
    enc_tensor = tf.constant([enc_ids], dtype=tf.int32)

    enc_seq, enc_state = encoder_model(enc_tensor)
    dec_input_token = tf.constant([[start_id]], dtype=tf.int32)
    generated_ids = []
    state = enc_state

    for step in range(max_dec_len):
        dec_logits, state = decoder_model([dec_input_token, state, enc_seq])
        logits = dec_logits[:, -1, :]
        if temperature==0.0:
            pred_id = tf.argmax(logits, axis=-1)
        else:
            pred_id = tf.random.categorical(logits/temperature,1)
            pred_id = tf.squeeze(pred_id, axis=-1)
        if int(pred_id[0])==end_id:
            break
        generated_ids.append(int(pred_id[0]))
        dec_input_token = tf.expand_dims(pred_id, axis=0)
        if verbose:
            print(f"Step {step}: ID={int(pred_id[0])}, Token='{sp.decode([int(pred_id[0])])}'")
    return sp.decode(generated_ids)

# =======================
# 테스트
# =======================
user_input = "오늘 날씨가 어때?"
response = generate(user_input, verbose=True)
print("\nAI 답변:", response)
