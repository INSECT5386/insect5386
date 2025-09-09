!pip install sentencepiece
import sentencepiece as spm

# 불러오기
import os, json, numpy as np, tensorflow as tf
import requests
print('1')

tf.get_logger().setLevel("ERROR")
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# TPU 초기화
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    print("✅ TPU 초기화 완료:", resolver.cluster_spec().as_dict())
    on_tpu = True
except Exception as e:
    print("⚠️ TPU 미사용, GPU/CPU로 진행:", e)
    strategy = tf.distribute.get_strategy()
    on_tpu = False

# Mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy("mixed_bfloat16" if on_tpu else "float32")
mixed_precision.set_global_policy(policy)
print("✅ Mixed precision:", policy)

# =======================
# 1) 파일 다운로드
# =======================
def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

DATA_PATH = "converted.jsonl"
TOKENIZER_PATH = "ko_unigram.model"

if not os.path.exists(DATA_PATH):
    download_file(
        "https://huggingface.co/datasets/Yuchan5386/SFT/resolve/main/data_shuffled_1.jsonl?download=true",
        DATA_PATH
    )

if not os.path.exists(TOKENIZER_PATH):
    download_file(
        "https://huggingface.co/Yuchan5386/inlam-100m/resolve/main/ko_unigram.model?download=true",
        TOKENIZER_PATH
    )

sp = spm.SentencePieceProcessor(TOKENIZER_PATH)

pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
start_id = sp.piece_to_id("<start>")
sep_id = sp.piece_to_id("<sep>")
end_id = sp.piece_to_id("<end>")
unk_id = sp.piece_to_id("<unk>")
vocab_size = sp.get_piece_size()
print(f"✅ Vocabulary size: {vocab_size}")

max_len = 1024
batch_size = 128

def text_to_ids(text):
    return sp.encode(text, out_type=int)
def ids_to_text(ids):
    return sp.decode(ids)

def jsonl_stream(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            conversations = data.get("conversations", [])
            for i in range(0, len(conversations) - 1, 2):
                human_msg = conversations[i]
                gpt_msg   = conversations[i + 1]
                if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                    continue
                prompt   = human_msg.get("value", "").strip()
                response = gpt_msg.get("value", "").strip()
                full = f"<start> {prompt} <sep> {response} <end>"
                if "<sep>" not in full:
                    continue
                sep_index  = full.index("<sep>")
                input_text = full[:sep_index + len("<sep>")].strip()
                target_text = full[sep_index + len("<sep>"):].strip()

                input_ids  = text_to_ids(input_text)
                target_ids = text_to_ids(target_text + " <end>")

                available_len = max_len - len(input_ids)
                if available_len <= 0:
                    input_ids = input_ids[-max_len:]
                    target_ids = []
                    target_mask = [0] * len(input_ids)
                else:
                    target_ids = target_ids[:available_len]
                    target_mask = [0] * len(input_ids) + [1] * len(target_ids)

                full_input = input_ids + target_ids
                pad_len = max_len - len(full_input)
                full_input += [pad_id] * pad_len
                target_mask += [0] * pad_len

                target_seq = full_input[1:] + [end_id]
                target_seq = target_seq[:max_len]

                masked_target = [
                    t if m == 1 else pad_id
                    for t, m in zip(target_seq, target_mask)
                ]

                yield (
                    tf.convert_to_tensor(full_input, dtype=tf.int32),
                    tf.convert_to_tensor(masked_target, dtype=tf.int32)
                )

dataset = tf.data.Dataset.from_generator(
    lambda: jsonl_stream(DATA_PATH),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
    ),
)
dataset = dataset.shuffle(1000, seed=SEED).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

import tensorflow as tf

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_model * expansion, dtype="bfloat16")
        self.out  = tf.keras.layers.Dense(d_model, dtype="bfloat16")

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))


class ContextAwareGate(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.query_proj = tf.keras.layers.Dense(d_model, dtype="bfloat16")
        self.ln = tf.keras.layers.LayerNormalization(dtype="bfloat16")
        self.scale = tf.Variable(0.1, dtype="bfloat16")  # 게이트 포화 방지 스케일

    def call(self, x, last_token):  # x: (B, S, D), last_token: (B, D)
        query_gate = self.query_proj(last_token)[:, tf.newaxis, :]  # (B, 1, D)
        combined = x + query_gate  # (B, S, D)
        combined = self.ln(combined)
        gate = tf.sigmoid(combined * self.scale)  # 포화 방지 스케일링
        return x * gate


class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(dtype="bfloat16")
        # ✅ 동적 시퀀스 길이 지원: EinsumDense로 공간 혼합 구현
        self.spatial_proj = tf.keras.layers.EinsumDense(
            equation="BSD,DS->BSD",
            output_shape=(d_model, max_seq_len),  # 가중치는 최대 길이 기준으로 학습
            dtype="bfloat16"
        )
        self.context_gate = ContextAwareGate(d_model)
        self.ln2 = tf.keras.layers.LayerNormalization(dtype="bfloat16")
        self.ffn = SwiGLU(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, last_token=None, training=False):
        seq_len = tf.shape(x)[1]

        # LayerNorm + Spatial Mixing
        y = self.ln1(x)
        y_t = tf.transpose(y, [0, 2, 1])  # (B, D, S)
        y_t = self.spatial_proj(y_t)       # (B, D, S_max) → 실제 S에 맞게 슬라이스
        y_t = y_t[:, :, :seq_len]          # ✅ 동적 길이에 맞춰 슬라이싱
        y = tf.transpose(y_t, [0, 2, 1])   # (B, S, D)

        # Context Gate: 학습 시에는 마지막에서 두 번째 토큰 사용 (미래 유출 방지)
        if last_token is None:
            if training:
                # 학습 시: teacher forcing → 마지막 토큰은 라벨이므로 사용 불가
                # 직전 토큰 사용 (예: i번째 위치에서는 i-1번째 토큰 기준 게이팅)
                last_token = x[:, -2, :] if seq_len > 1 else x[:, 0, :]
            else:
                # 추론 시: 마지막 토큰 사용
                last_token = x[:, -1, :]

        y = self.context_gate(y, last_token)

        x = x + self.dropout(y, training=training)

        y = self.ln2(x)
        x = x + self.dropout(self.ffn(y), training=training)
        return x


class InLaM(tf.keras.Model):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, dropout_rate=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, dtype="bfloat16")
        self.pos_embedding = tf.keras.layers.Embedding(max_seq_len, d_model, dtype="bfloat16")
        # ✅ gMLPBlock도 max_seq_len 전달
        self.blocks = [gMLPBlock(d_model, max_seq_len, dropout_rate) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype="bfloat16")

    def call(self, x, last_token=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # ✅ 동적 포지셔널 임베딩
        positions = tf.range(seq_len)[tf.newaxis, :]  # (1, S)
        positions = tf.clip_by_value(positions, 0, self.max_seq_len - 1)  # 안전장치
        pos_embed = self.pos_embedding(positions)  # (1, S, D)
        x = self.token_embedding(x) + pos_embed    # (B, S, D)

        # ✅ 각 블록에 last_token 전달
        for block in self.blocks:
            x = block(x, last_token=last_token, training=training)

        # ✅ 로짓 계산 전 float32로 캐스팅 (정밀도 보존)
        x = tf.cast(self.ln_f(x), tf.float32)
        embed_weights = tf.cast(self.token_embedding.weights[0], tf.float32)
        logits = tf.matmul(x, embed_weights, transpose_b=True)  # (B, S, V)
        return logits

# =======================
def smoothed_loss_keras(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1)
    per_tok = per_tok * mask
    return tf.reduce_sum(per_tok) / (tf.reduce_sum(mask) + 1e-8)

def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    pred_id = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true, pred_id), tf.float32) * mask
    return tf.reduce_sum(acc) / (tf.reduce_sum(mask) + 1e-8)

def masked_perplexity(y_true, y_pred, eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true, depth=vocab, dtype=tf.float32)
    y_true_ls = (1.0 - eps) * y_true_oh + eps / tf.cast(vocab, tf.float32)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls * log_probs, axis=-1)
    per_tok = per_tok * mask
    mean_loss = tf.reduce_sum(per_tok) / (tf.reduce_sum(mask) + 1e-8)
    return tf.exp(mean_loss)


# =======================
# 모델 생성 & 컴파일
# =======================
with strategy.scope():
    model = InLaM(vocab_size, seq_len=max_len, d_model=512, n_layers=8)
    dummy_input = tf.zeros((batch_size, max_len), dtype=tf.int32)
    _ = model(dummy_input, training=False)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.95, epsilon=1e-8, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=smoothed_loss_keras, metrics=[masked_accuracy, masked_perplexity])

    # 학습
    history = model.fit(dist_dataset, epochs=1, verbose=1)

# =======================
# 가중치 저장
# =======================
model.save_weights("tf_model.weights.h5")
print("✅ 모델 가중치 저장 완료!")

# =======================
# 샘플 생성 함수
# =======================
def generate_text_topp(model, prompt, max_len=128, max_gen=98, p=0.9, temperature=0.68, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    
    for step in range(max_gen):
        input_seq = generated[-max_len:] if len(generated) > max_len else generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded], dtype=tf.int32)
        
        logits = model(input_tensor, training=False).numpy()[0, len(input_seq)-1]
        logits[end_id] -= 5.0
        logits[pad_id] -= 10.0
        
        probs = tf.nn.softmax(logits / temperature).numpy()
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, p)
        top_idx = sorted_idx[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1] / sorted_probs[:cutoff + 1].sum()
        
        next_token = int(np.random.choice(top_idx, p=top_probs))
        if next_token == end_id and len(generated) >= min_len:
            break
        generated.append(next_token)
    
    return ids_to_text(generated)

# =======================
# 테스트 생성
# =======================
prompt = "딥러닝에 대해 설명하세요."
sample_text = generate_text_topp(model, prompt, p=0.9)
print("\n===== 생성 결과 =====\n")
print(sample_text)




