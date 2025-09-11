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

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, f_d=8//3):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_model*f_d)
        self.out  = tf.keras.layers.Dense(d_model)
    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))

class Block(tf.keras.layers.Layer):
    def __init__(self, d_model, seq_len, expand_ratio=2, kernel_size=15, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * expand_ratio
        
        # ✅ 지역 문맥 강화 — Depthwise Conv1D (causal!)
        self.conv = tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=kernel_size,
            padding='causal',  # 👈 핵심 변경!
            groups=d_model,
            activation=None,
            name="local_context_conv"
        )
        
        # ✅ 전역 문맥 — 확장된 공간 게이팅
        self.proj_up = tf.keras.layers.Dense(self.hidden_dim, name="expand_channel")
        self.spatial_gate = tf.keras.layers.Dense(seq_len, use_bias=True, name="spatial_gate")
        self.spatial_proj = tf.keras.layers.Dense(seq_len, use_bias=False, name="spatial_proj")
        self.proj_down = tf.keras.layers.Dense(d_model, name="compress_channel")
        
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = SwiGLU(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        y = self.ln1(x)
        y = self.conv(y)  # (B, S, D) — 이제 미래 정보 차단됨!
        y = self.proj_up(y)  # (B, S, D*expand)
        y_t = tf.transpose(y, [0, 2, 1])  # (B, D*expand, S)
        gate = tf.nn.silu(self.spatial_gate(y_t))
        y_t = self.spatial_proj(y_t) * gate
        y = tf.transpose(y_t, [0, 2, 1])  # (B, S, D*expand)
        y = self.proj_down(y)  # (B, S, D)
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
        self.blocks = [Block(d_model, seq_len=max_seq_len, dropout_rate=0.1) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype="bfloat16")

    def call(self, x, last_token=None, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # 동적 포지셔널 임베딩
        positions = tf.range(seq_len)[tf.newaxis, :]  # (1, S)
        positions = tf.clip_by_value(positions, 0, self.max_seq_len - 1)
        pos_embed = self.pos_embedding(positions)     # (1, S, D)
        x = self.token_embedding(x) + pos_embed       # (B, S, D)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.ln_f(x)
        # Output logits
        embed_weights = self.token_embedding.weights[0]
        logits = tf.matmul(x, embed_weights, transpose_b=True)
        return tf.cast(logits, tf.float32)
# 손실/메트릭 정의
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
    model = InLaM(vocab_size, max_seq_len=max_len, d_model=384, n_layers=16, dropout_rate=0.1)
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
def generate_text_topp(model, prompt, max_len=1024, max_gen=200, p=0.9, temperature=0.68, min_len=20):
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



