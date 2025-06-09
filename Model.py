import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers  
import sentencepiece as spm  
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
download_file('https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/dataset.parquet?download=true', 'dataset.parquet')
download_file('https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/kolig_unigram.model?download=true', 'ko_unigram.model')

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
train_sentences = train_sentences[:10000]
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
max_len = 100
batch_size = 42

# ⬇️ 인풋과 타겟 마스킹 포함된 전처리
encoded_inputs = []
targets = []

for sentence in train_sentences:
    if "<sep>" not in sentence:
        continue

    sep_index = sentence.index("<sep>")
    input_text = sentence[:sep_index + len("<sep>")].strip()
    target_text = sentence[sep_index + len("<sep>"):].strip()

    input_ids = text_to_ids(input_text)
    target_ids = text_to_ids(target_text + " <end>")

    full_input = input_ids + target_ids
    full_input = full_input[:max_len]

    target_mask = [0] * len(input_ids) + [1] * len(target_ids)
    target_mask = target_mask[:max_len]

    if len(full_input) < max_len:
        pad_len = max_len - len(full_input)
        full_input += [pad_id] * pad_len
        target_mask += [0] * pad_len

    encoded_inputs.append(full_input)

    target_seq = full_input[1:] + [end_id]
    target_seq = target_seq[:max_len]

    masked_target = [
        t if m == 1 else pad_id
        for t, m in zip(target_seq, target_mask)
    ]

    targets.append(masked_target)

# ⬇️ 넘파이 변환
encoded_inputs = np.array(encoded_inputs)
targets = np.array(targets)

# ⬇️ TensorFlow Dataset 생성
def data_generator():
    for input_seq, target_seq in zip(encoded_inputs, targets):
        yield input_seq, target_seq

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)

dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("✅ TF Dataset 생성 완료!")

class RotaryPositionalEmbedding(layers.Layer):  
    def __init__(self, dim):  
        super().__init__()  
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))  
        self.inv_freq = tf.constant(inv_freq, dtype=tf.float32)  
  
    def call(self, x):  
        batch, heads, seq_len, depth = tf.unstack(tf.shape(x))  
        t = tf.range(seq_len, dtype=tf.float32)  
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)  
        emb_sin = tf.sin(freqs)  
        emb_cos = tf.cos(freqs)  
        emb_cos = tf.reshape(emb_cos, [1, 1, seq_len, -1])  
        emb_sin = tf.reshape(emb_sin, [1, 1, seq_len, -1])  
        x1 = x[..., ::2]  
        x2 = x[..., 1::2]  
        x_rotated = tf.stack([  
            x1 * emb_cos - x2 * emb_sin,  
            x1 * emb_sin + x2 * emb_cos  
        ], axis=-1)  
        x_rotated = tf.reshape(x_rotated, tf.shape(x))  
        return x_rotated

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff * 2)
        self.out = tf.keras.layers.Dense(d_model)

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))
        
class Block(tf.keras.layers.Layer):  
    def __init__(self, d_model, d_ff, num_heads=4, dropout_rate=0.05, adapter_dim=48):    
        super().__init__()    
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)    
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)    
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)   
        self.adapter_down = tf.keras.layers.Dense(adapter_dim, activation='gelu')   
        self.adapter_up = tf.keras.layers.Dense(d_model)    
    
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)    
        self.ffn = SwiGLU(d_model, d_ff)    
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)   
        self.rope = RotaryPositionalEmbedding(d_model // num_heads)    
    
    def call(self, x, training=False):    
        x_norm = self.ln1(x)    
        b, s, _ = tf.shape(x_norm)[0], tf.shape(x_norm)[1], tf.shape(x_norm)[2]    
        h = self.mha.num_heads    
        d = x_norm.shape[-1] // h    
    
        qkv = tf.reshape(x_norm, [b, s, h, d])    
        qkv = tf.transpose(qkv, [0, 2, 1, 3])    
        q = self.rope(qkv)    
        k = self.rope(qkv)    
        q = tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [b, s, h * d])    
        k = tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [b, s, h * d])    
    
        attn_out = self.mha(query=q, value=x_norm, key=k, use_causal_mask=True, training=training)    
        attn_out = self.dropout1(attn_out, training=training)    
  
        adapter_out = self.adapter_up(self.adapter_down(attn_out))  
        attn_out = attn_out + adapter_out    
    
        x = x + attn_out    
        ffn_out = self.ffn(self.ln2(x))    
        x = x + self.dropout2(ffn_out, training=training)    
        return x

class Flexi(tf.keras.Model):  
    def __init__(self, vocab_size, seq_len, d_model, d_ff, n_layers, num_heads=4, dropout_rate=0.1):  
        super().__init__()  
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)  
        self.blocks = [Block(d_model, d_ff, num_heads, dropout_rate) for _ in range(n_layers)]  
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)  
  
    def call(self, x, training=False):  
        x = self.token_embedding(x)  
        for block in self.blocks:  
            x = block(x, training=training)  
        x = self.ln_f(x)  
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)  
        return logits  

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def masked_loss(y_true, y_pred):  
    loss = loss_fn(y_true, y_pred)  
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)  
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask) 

model = Flexi(
    vocab_size=vocab_size,
    seq_len=max_len,
    d_model=128,
    d_ff=512,       
    n_layers=13
)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss=masked_loss)
dummy_input = np.zeros((1, max_len), dtype=np.int32)

model(dummy_input)  
model.summary()

steps_per_epoch = len(encoded_inputs) // batch_size
model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch)
model.save_weights("Flexi.weights.h5")  
print("모델 가중치 저장 완료!")  

def generate_text_topp(model, prompt, max_len=100, max_gen=98, p=0.9, temperature=0.8, min_len=20):
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    for step in range(max_gen):
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated
        input_padded = np.pad(input_seq, (0, max_len - len(input_seq)), constant_values=pad_id)
        input_tensor = tf.convert_to_tensor([input_padded])
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        next_token_logits[end_id] -= 5.0
        next_token_logits[pad_id] -= 10.0
        probs = tf.nn.softmax(next_token_logits / temperature).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        next_token_id = np.random.choice(top_indices, p=top_probs)
        if next_token_id == end_id and len(generated) >= min_len:
            break
        generated.append(int(next_token_id))
    return ids_to_text(generated)

print("\n\n===== 생성 결과 =====")  
print(generate_text_topp(model, "안녕", p=0.9))
