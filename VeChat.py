import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import sentencepiece as spm
import json
import requests

# =======================
# 0) 파일 다운로드
# =======================
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

download_file('https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/dataset.jsonl?download=true', 'VeTrans.jsonl')
download_file('https://huggingface.co/datasets/Yuchan5386/Test1111/resolve/main/kolig_unigram.model?download=true', 'ko_unigram.model')

# =======================
# 1) Tokenizer
# =======================
sp = spm.SentencePieceProcessor()
sp.load('ko_unigram.model')
vocab_size = sp.get_piece_size()

pad_id = sp.piece_to_id("<pad>") or 0
start_id = sp.piece_to_id("<start>") or sp.bos_id() or 1
end_id = sp.piece_to_id("<end>") or sp.eos_id() or 2

print("TOKENS:", {"pad": pad_id, "start": start_id, "end": end_id, "vocab": vocab_size})

max_len = 100
batch_size = 64

# =======================
# 2) Data generator
# =======================
def data_generator(file_path, max_len=100, pad_id=0, start_id=1, end_id=2, max_samples=None):
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples and count >= max_samples:
                break
            item = json.loads(line)
            if "conversations" not in item or len(item["conversations"]) < 2:
                continue
            src = item["conversations"][0].get("value","").strip()
            tgt = item["conversations"][1].get("value","").strip()
            if len(src) < 2 or len(tgt) < 2 or src==tgt:
                continue
            def encode_text(text):
                ids = sp.encode(text, out_type=int)
                ids = [start_id]+ids+[end_id]
                if len(ids)<max_len:
                    ids += [pad_id]*(max_len-len(ids))
                else:
                    ids = ids[:max_len]
                    if ids[-1]!=end_id: ids[-1]=end_id
                return np.array(ids,dtype=np.int32)
            yield encode_text(src), encode_text(tgt)
            count += 1

stream_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator("VeTrans.jsonl", max_len=max_len, pad_id=pad_id, start_id=start_id, end_id=end_id, max_samples=500000),
    output_signature=(
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
    )
)
stream_dataset = stream_dataset.shuffle(5000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def map_fn(src, tgt):
    dec_input = tgt[:, :-1]
    dec_target = tgt[:, 1:]
    return ({"enc_inputs": src, "dec_inputs": dec_input}, dec_target)

train_ds = stream_dataset.map(map_fn)

# =======================
# 3) SwiGLU FFN
# =======================
class SwiGLU(layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = layers.Dense(d_ff*2)
        self.out = layers.Dense(d_model)
    def call(self, x):
        x_val, x_gate = tf.split(self.proj(x), 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))

# =======================
# 4) SSM 레이어
# =======================
class SSMBlock(layers.Layer):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.A = self.add_weight(shape=(d_model,), initializer=tf.keras.initializers.RandomNormal(-0.5,0.1), trainable=True)
        self.B = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True)
        self.C = self.add_weight(shape=(d_model,), initializer='random_normal', trainable=True)
        self.D = self.add_weight(shape=(d_model,), initializer='zeros', trainable=True)
        self.ffn = SwiGLU(d_model, d_ff)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
    def call(self, x, training=False):
        # gating
        u = x * tf.nn.sigmoid(x)
        B = tf.shape(x)[0]; T = tf.shape(x)[1]; D = self.d_model
        t_idx = tf.cast(tf.range(T), x.dtype)[:, None]  # (T,1)
        A_pow = tf.pow(tf.expand_dims(self.A,0), t_idx)  # (T,D)
        y_states = u * A_pow[None,:,:] * self.B[None,None,:]  # (B,T,D)
        y = y_states * self.C[None,None,:] + self.D[None,None,:]*u  # (B,T,D)
        y = self.norm(y)
        y = self.ffn(y)
        y = self.dropout(y, training=training)
        return y

# =======================
# 5) SSMDecoderBlock
# =======================
class SSMDecoderBlock(layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout=0.1):
        super().__init__()
        self.ssm = SSMBlock(d_model, d_ff=d_ff, dropout=dropout)
        self.cross_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = SwiGLU(d_model, d_ff)
    def call(self, dec_emb, enc_out, training=False):
        x = self.ssm(dec_emb, training=training)
        attn_out = self.cross_mha(x, enc_out, enc_out)
        x = x + attn_out
        x = self.ffn(x)
        out = self.norm(x)
        return out

# =======================
# 6) Transformer
# =======================
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size,
                 max_len=100, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # Embedding
        self.enc_embedding = layers.Embedding(input_vocab_size,d_model)
        self.enc_pos_embedding = layers.Embedding(max_len,d_model)
        self.dec_embedding = layers.Embedding(target_vocab_size,d_model)
        self.dec_pos_embedding = layers.Embedding(max_len,d_model)
        # Layers
        self.enc_layers = [SSMBlock(d_model, d_ff=512, dropout=0.1) for _ in range(num_layers)]
        self.dec_layers = [SSMDecoderBlock(d_model,d_ff=dff,num_heads=num_heads,dropout=dropout) for _ in range(num_layers)]
        self.final_layer = layers.Dense(target_vocab_size)
    def call(self, inputs, training=False):
        enc_inputs = inputs["enc_inputs"]
        dec_inputs = inputs["dec_inputs"]
        enc_pos = tf.range(tf.shape(enc_inputs)[1])[tf.newaxis,:]
        dec_pos = tf.range(tf.shape(dec_inputs)[1])[tf.newaxis,:]
        # Encoder
        x = self.enc_embedding(enc_inputs)+self.enc_pos_embedding(enc_pos)
        for layer in self.enc_layers:
            x = layer(x, training=training)
        enc_out = x
        # Decoder
        y = self.dec_embedding(dec_inputs)+self.dec_pos_embedding(dec_pos)
        for layer in self.dec_layers:
            y = layer(y, enc_out, training=training)
        logits = self.final_layer(y)
        return logits

model = Transformer(num_layers=4,d_model=128,num_heads=8,dff=512,
                    input_vocab_size=vocab_size,target_vocab_size=vocab_size)

# =======================
# 7) Loss & Metrics
# =======================
optimizer = tf.keras.optimizers.Adam(1e-4, clipnorm=1.0)

def smoothed_loss_keras(y_true,y_pred,eps=0.1):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true,pad_id), tf.float32)
    vocab = tf.shape(y_pred)[-1]
    y_true_oh = tf.one_hot(y_true,depth=vocab,dtype=tf.float32)
    y_true_ls = (1.0-eps)*y_true_oh + eps/tf.cast(vocab,tf.float32)
    log_probs = tf.nn.log_softmax(y_pred,axis=-1)
    per_tok = -tf.reduce_sum(y_true_ls*log_probs,axis=-1)
    per_tok = per_tok*mask
    return tf.reduce_sum(per_tok)/(tf.reduce_sum(mask)+1e-8)

def masked_accuracy(y_true,y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true,pad_id), tf.float32)
    pred_id = tf.argmax(y_pred,axis=-1,output_type=tf.int32)
    acc = tf.cast(tf.equal(y_true,pred_id), tf.float32)*mask
    return tf.reduce_sum(acc)/(tf.reduce_sum(mask)+1e-8)

tf.config.optimizer.set_jit(True)
model.compile(optimizer=optimizer, loss=smoothed_loss_keras, metrics=[masked_accuracy], run_eagerly=False)
model.summary()
# =======================
# 8) Training
# =======================
model.fit(train_ds, epochs=1)
model.save_weights("VeChat_SSM.weights.h5")
print("✅ 모델 가중치 저장 완료!")

# =======================
# 9) Top-p Sampling
# =======================
def top_p_sample(logits,p=0.9):
    sorted_ids = tf.argsort(logits,direction='DESCENDING')
    sorted_logits = tf.gather(logits,sorted_ids)
    probs = tf.nn.softmax(sorted_logits)
    cumulative_probs = tf.cumsum(probs)
    cutoff = tf.reduce_sum(tf.cast(cumulative_probs<=p, tf.float32))
    top_ids = sorted_ids[:tf.cast(cutoff+1,tf.int32)]
    top_logits = tf.gather(logits,top_ids)
    top_probs = tf.nn.softmax(top_logits)
    return np.random.choice(top_ids.numpy(),p=top_probs.numpy())

def generate_text(src_text,max_len=100,top_p=0.9):
    def encode_text(text):
        ids = sp.encode(text,out_type=int)
        ids = [start_id]+ids+[end_id]
        if len(ids)<max_len:
            ids += [pad_id]*(max_len-len(ids))
        else:
            ids = ids[:max_len]
            if ids[-1]!=end_id: ids[-1]=end_id
        return ids
    src_ids = np.array([encode_text(src_text)])
    enc_emb = model.enc_embedding(src_ids)
    for layer in model.enc_layers:
        enc_emb = layer(enc_emb, training=False)
    out_seq = [start_id]
    for _ in range(max_len-1):
        dec_ids = np.array([out_seq])
        dec_emb = model.dec_embedding(dec_ids)
        for layer in model.dec_layers:
            dec_emb = layer(dec_emb, enc_emb, training=False)
        logits = model.final_layer(dec_emb)[0,-1]
        next_id = top_p_sample(logits,top_p)
        if next_id==end_id: break
        out_seq.append(next_id)
    toks = [int(t) for t in out_seq if t not in (pad_id,start_id,end_id)]
    return sp.decode(toks)

# =======================
# 10) Test
# =======================
print(generate_text("안녕하세요!"))
print(generate_text("오늘은 뭐 할 거야?"))
print(generate_text("오늘 기분은 어때?"))
