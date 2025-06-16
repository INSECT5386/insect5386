import json  
import numpy as np  
import pandas as pd
import tensorflow as tf  
from tensorflow.keras import layers 
import sentencepiece as spm  
import requests
import math

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

train_sentences = train_sentences[:300000]
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

# ⬇️ 개선된 하이퍼파라미터
max_len = 256  # 증가된 시퀀스 길이
batch_size = 32  # 조정된 배치 크기
d_model = 384  # 증가된 모델 차원
n_layers = 12  # 증가된 레이어 수
state_size = 16  # Mamba state 크기

# ⬇️ 개선된 데이터 전처리 함수
def create_efficient_dataset(sentences, batch_size, max_len):
    """메모리 효율적인 데이터셋 생성"""
    
    def preprocess_batch(batch_sentences):
        batch_inputs = []
        batch_targets = []
        batch_masks = []
        
        for sentence in batch_sentences:
            if "<sep>" not in sentence:
                continue
                
            sep_index = sentence.index("<sep>")
            input_text = sentence[:sep_index + len("<sep>")].strip()
            target_text = sentence[sep_index + len("<sep>"):].strip()
            
            input_ids = text_to_ids(input_text)
            target_ids = text_to_ids(target_text + " <end>")
            
            full_input = input_ids + target_ids
            full_input = full_input[:max_len]
            
            # 동적 마스킹
            target_mask = [0] * len(input_ids) + [1] * len(target_ids)
            target_mask = target_mask[:max_len]
            
            # 패딩
            if len(full_input) < max_len:
                pad_len = max_len - len(full_input)
                full_input += [pad_id] * pad_len
                target_mask += [0] * pad_len
            
            target_seq = full_input[1:] + [end_id]
            target_seq = target_seq[:max_len]
            
            masked_target = [
                t if m == 1 else pad_id
                for t, m in zip(target_seq, target_mask)
            ]
            
            batch_inputs.append(full_input)
            batch_targets.append(masked_target)
            batch_masks.append(target_mask)
        
        return (
            tf.constant(batch_inputs, dtype=tf.int32),
            tf.constant(batch_targets, dtype=tf.int32),
            tf.constant(batch_masks, dtype=tf.float32)
        )
    
    # 배치 단위로 처리
    dataset = tf.data.Dataset.from_tensor_slices(sentences)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x: tf.py_function(
            preprocess_batch, [x], 
            [tf.int32, tf.int32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)


# ======================= 개선된 SwiGLU FFN ======================
class SwiGLUFFN(tf.keras.layers.Layer):
    """SwiGLU activation을 사용한 개선된 FFN"""
    def __init__(self, dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)  # SwiGLU 표준 비율
        
        self.gate_proj = layers.Dense(hidden_dim, use_bias=False)
        self.up_proj = layers.Dense(hidden_dim, use_bias=False)
        self.down_proj = layers.Dense(dim, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = tf.nn.silu(gate) * up
        hidden = self.dropout(hidden, training=training)
        return self.down_proj(hidden)

import tensorflow as tf
from tensorflow.keras import layers

class ImprovedMambaCore(tf.keras.layers.Layer):
    """tf.scan을 사용한 더 효율적인 구현"""
    def __init__(self, hidden_dim, state_size=16, conv_kernel=4, expand_factor=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.expand_factor = expand_factor
        self.inner_dim = hidden_dim * expand_factor
        
        # Input projections
        self.in_proj = layers.Dense(self.inner_dim * 2, use_bias=False)
        
        # Convolution layer
        self.conv1d = layers.Conv1D(
            filters=self.inner_dim,
            kernel_size=conv_kernel,
            padding='same',
            use_bias=True
        )
        
        # SSM parameters projection
        self.x_proj = layers.Dense(state_size * 2, use_bias=False)
        self.dt_proj = layers.Dense(self.inner_dim, use_bias=True)
        
        # SSM parameters
        self.A_log = self.add_weight(
            shape=(self.inner_dim, state_size),
            initializer=lambda shape, dtype: tf.math.log(
                tf.random.uniform(shape, 1, 16, dtype=dtype)
            ),
            trainable=True,
            name="A_log"
        )
        
        self.D = self.add_weight(
            shape=(self.inner_dim,),
            initializer='ones',
            trainable=True,
            name="D"
        )
        
        # Output projection
        self.out_proj = layers.Dense(hidden_dim, use_bias=False)
        
        # Normalization
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def selective_scan_v2(self, x, delta, A_log_param, B, C, D): # A_log는 self.A_log와 이름 충돌 방지
        """tf.scan을 사용한 selective scan"""
        batch_size = tf.shape(x)[0]
        
        # Discretization
        delta = tf.nn.softplus(delta)
        # A는 self.A_log를 사용해야 함
        A = -tf.exp(A_log_param)  # (inner_dim, state_size)
        
        # Discretize A and B (einsum 순서 변경: (batch, seq, inner_dim, state_size)로 만들기 위함)
        # tf.einsum('bld,dn->bldn', delta, A)에서 delta는 (b,l,d), A는 (d,n)이므로 결과는 (b,l,d,n)
        # 여기서는 A_discrete의 (batch, seq, inner_dim, state_size) 형태가 필요합니다.
        # A는 (inner_dim, state_size) 이므로, delta(batch, seq, inner_dim)와 연산하려면
        # delta를 (batch, seq, inner_dim, 1)으로 확장하여 A를 브로드캐스트 할 수 있습니다.
        # 그러나 실제 Mamba의 Selective Scan 구현에서는 A가 x와 독립적으로 작동합니다.
        # A_discrete는 A (inner_dim, state_size)와 delta (batch, seq, inner_dim)의 관계를 표현해야 합니다.
        # Mamba의 A는 시퀀스 길이에 따라 변하지 않는 고정된 파라미터입니다.
        # A_discrete = tf.exp(tf.einsum('bld,dn->bldn', delta, A))
        # 위 라인은 A를 (inner_dim, state_size)에서 (b,l,inner_dim, state_size)로 확장하려는 시도인데,
        # A는 시퀀스 차원(l)을 가지지 않습니다.
        # Mamba에서 A는 (D, N) 형태이며, delta (B, L, D)와 결합하여 A_bar = exp(delta * A) 가 됩니다.
        
        # A_discrete 계산 수정: (batch, seq, inner_dim, state_size)
        # A_log_param: (inner_dim, state_size)
        # delta: (batch, seq, inner_dim)
        # A_discrete = exp(delta * A)
        # tf.einsum('bld,dn->bldn', delta, A)는 delta의 마지막 차원과 A의 첫 차원을 곱합니다.
        # 이는 A가 (inner_dim, state_size)이고 delta가 (batch, seq, inner_dim)일 때 적합하지 않습니다.
        # 올바른 A_discrete는 (batch, seq, inner_dim, state_size) 형태여야 합니다.
        # Mamba의 Selective Scan의 A_bar는 delta * A_prime 이고, A_prime은 A의 파생 버전입니다.
        # 여기서는 `A`가 이미 `self.A_log`에서 온 `(inner_dim, state_size)` 형태이므로,
        # `delta`와 `A`를 직접 곱하려면 차원을 맞춰줘야 합니다.
        # delta를 (batch, seq, inner_dim, 1)으로 확장하고, A를 (1, 1, inner_dim, state_size)로 확장하여 브로드캐스트 곱셈을 수행합니다.
        A_expanded = tf.expand_dims(A, axis=0) # (1, inner_dim, state_size)
        A_expanded = tf.expand_dims(A_expanded, axis=0) # (1, 1, inner_dim, state_size)
        delta_expanded = tf.expand_dims(delta, axis=-1) # (batch, seq, inner_dim, 1)
        A_discrete = tf.exp(delta_expanded * A_expanded) # (batch, seq, inner_dim, state_size)

        # B_discrete 계산 수정: (batch, seq, inner_dim, state_size)
        # B는 (batch, seq, state_size)
        # delta는 (batch, seq, inner_dim)
        # B_discrete = delta * B
        # tf.einsum('bld,bln->bldn', delta, B)는 올바르지 않습니다.
        # delta는 (batch, seq, inner_dim), B는 (batch, seq, state_size)
        # Mamba의 B_bar는 delta * B 이며, B는 (B, L, N) 형태입니다.
        # 따라서 delta와 B를 브로드캐스트 곱셈하기 위해 차원을 맞춰야 합니다.
        # delta_expanded는 (batch, seq, inner_dim, 1)
        # B_expanded = tf.expand_dims(B, axis=-2) # (batch, seq, 1, state_size)
        # B_discrete = delta_expanded * B_expanded # (batch, seq, inner_dim, state_size)
        # Mamba 오리지널 구현에 가깝게, B는 (batch, seq, state_size)이고 이를 inner_dim 만큼 반복하여
        # (batch, seq, inner_dim, state_size) 형태로 만듭니다.
        # tf.einsum('bld,bln->bldn', delta, B) 대신 `tf.tile`을 사용하거나 차원을 조정하여 곱해야 합니다.
        # 여기서는 B의 차원이 이미 (batch, seq, state_size)이므로, delta와 브로드캐스트 하려면
        # delta (batch, seq, inner_dim)를 (batch, seq, inner_dim, 1)으로 확장하고
        # B (batch, seq, state_size)를 (batch, seq, 1, state_size)으로 확장하여
        # 곱한 후 inner_dim 차원으로 다시 확장하는 형태가 되어야 합니다.
        # B_discrete = tf.einsum('bln,bld->bldn', B, delta) -> B의 차원 (b,l,n), delta의 차원 (b,l,d) 이므로 (b,l,d,n)

        # Mamba 구현 방식에 따라 B_discrete 계산:
        # B (batch, seq, state_size)를 inner_dim으로 확장
        B_expanded_for_mul = tf.expand_dims(B, axis=-2) # (batch, seq, 1, state_size)
        # delta (batch, seq, inner_dim)를 (batch, seq, inner_dim, 1)으로 확장
        delta_expanded_for_mul = tf.expand_dims(delta, axis=-1) # (batch, seq, inner_dim, 1)
        B_discrete = delta_expanded_for_mul * B_expanded_for_mul # (batch, seq, inner_dim, state_size)


        def scan_fn(carry, inputs):
            h_prev = carry
            x_t, A_t, B_t = inputs # x_t는 (batch, inner_dim)
            # A_t는 (batch, inner_dim, state_size)
            # B_t는 (batch, inner_dim, state_size)
            
            # h_new = A_t * h_prev + B_t * tf.expand_dims(x_t, -1)
            # h_prev: (batch, inner_dim, state_size)
            # x_t: (batch, inner_dim) -> tf.expand_dims(x_t, -1): (batch, inner_dim, 1)
            # B_t * tf.expand_dims(x_t, -1) : (batch, inner_dim, state_size) * (batch, inner_dim, 1) = (batch, inner_dim, state_size)
            # A_t * h_prev : (batch, inner_dim, state_size) * (batch, inner_dim, state_size) = (batch, inner_dim, state_size)
            h_new = A_t * h_prev + B_t * tf.expand_dims(x_t, -1)
            
            return h_new # <--- 이제 단일 텐서만 반환합니다.
        
        # Prepare inputs for scan (transpose to put sequence dimension first)
        x_scan = tf.transpose(x, [1, 0, 2])  # (seq, batch, inner_dim)
        # A_discrete, B_discrete의 shape: (batch, seq, inner_dim, state_size)
        A_scan = tf.transpose(A_discrete, [1, 0, 2, 3])  # (seq, batch, inner_dim, state_size)
        B_scan = tf.transpose(B_discrete, [1, 0, 2, 3])  # (seq, batch, inner_dim, state_size)
        
        # Initial state (batch, inner_dim, state_size)
        initial_state = tf.zeros([batch_size, self.inner_dim, self.state_size], dtype=x.dtype)
        
        # Run scan
        # tf.scan은 scan_fn이 반환하는 첫 번째 값을 다음 carry로 사용하고,
        # 반환하는 모든 값을 모아 최종 출력을 만듭니다.
        # 따라서 scan_fn이 h_new 하나만 반환하면, final_state는 마지막 h_new가 되고,
        # states는 모든 h_new를 모은 텐서가 됩니다.
        states = tf.scan(
            scan_fn,
            (x_scan, A_scan, B_scan),
            initializer=initial_state
        )
        
        # Transpose back to (batch, seq, inner_dim, state_size)
        states = tf.transpose(states, [1, 0, 2, 3])
        
        # Output computation
        # y = tf.einsum('blds,bls->bld', states, C)
        # C의 차원: (batch, seq, state_size)
        # states의 차원: (batch, seq, inner_dim, state_size)
        # 'blds,bls->bld'는 (batch, seq, inner_dim, state_size) * (batch, seq, state_size) -> (batch, seq, inner_dim)
        # 이는 올바른 einsum입니다.
        y = tf.einsum('blds,bls->bld', states, C)
        
        # D는 (inner_dim,) 이므로 x (batch, seq, inner_dim)와 브로드캐스트 곱셈
        y = y + D * x # D는 (inner_dim,)이므로 x (batch, seq, inner_dim)와 브로드캐스트됨
        
        return y

    def call(self, x):
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq, inner_dim * 2)
        x_inner, z = tf.split(xz, 2, axis=-1)
        
        # Convolution
        x_conv = self.conv1d(x_inner)
        x_conv = tf.nn.silu(x_conv)
        
        # SSM parameters
        x_ssm = self.x_proj(x_conv)  # (batch, seq, state_size * 2)
        B, C = tf.split(x_ssm, 2, axis=-1)  # Each (batch, seq, state_size)
        
        # Delta projection
        delta = self.dt_proj(x_conv)  # (batch, seq, inner_dim)
        
        # Selective scan
        y = self.selective_scan_v2(
            x_conv, delta, self.A_log, B, C, self.D
        )
        
        # Gate and output projection
        y = y * tf.nn.silu(z)
        output = self.out_proj(y)
        
        return self.norm(output)


# ======================= 개선된 Cobrablock ======================
class ImprovedCobrablock(tf.keras.layers.Layer):
    """개선된 Cobra 블록 with RMSNorm and better residuals"""
    def __init__(self, d_model, dropout_rate=0.1, use_rms_norm=True):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        
        if use_rms_norm:
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = layers.LayerNormalization(epsilon=1e-5)
            self.norm2 = layers.LayerNormalization(epsilon=1e-5)
            
        self.mamba = ImprovedMambaCore(d_model)
        self.ffn = SwiGLUFFN(d_model, dropout_rate=dropout_rate)
        
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Mamba block with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = residual + self.dropout1(x, training=training)
        
        # FFN block with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, training=training)
        x = residual + self.dropout2(x, training=training)
        
        return x


# ======================= RMSNorm Implementation ======================
class RMSNorm(tf.keras.layers.Layer):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = self.add_weight(
            shape=(dim,),
            initializer='ones',
            trainable=True,
            name='weight'
        )

    def call(self, x):
        norm = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return x / norm * self.weight


# ======================= 개선된 CobraModel ======================
class ImprovedCobraModel(tf.keras.Model):
    """개선된 Cobra 모델"""
    def __init__(self, vocab_size, d_model, n_layers, dropout_rate=0.1, use_rms_norm=True):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = layers.Embedding(vocab_size, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = self.add_weight(
            shape=(max_len, d_model),
            initializer='random_normal',
            trainable=True,
            name='pos_embedding'
        )
        
        self.blocks = [
            ImprovedCobrablock(d_model, dropout_rate, use_rms_norm) 
            for _ in range(n_layers)
        ]
        
        if use_rms_norm:
            self.ln_f = RMSNorm(d_model)
        else:
            self.ln_f = layers.LayerNormalization(epsilon=1e-5)
            
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        
        # Token + positional embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding[:seq_len]
        x = token_emb + pos_emb
        x = self.dropout(x, training=training)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection (tied embeddings)
        logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)
        return logits


# ======================= 개선된 손실 함수 및 메트릭 ======================
class ImprovedMetrics:
    @staticmethod
    def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        p_t = tf.exp(-ce_loss)
        focal_loss = alpha * tf.pow(1 - p_t, gamma) * ce_loss
        return focal_loss
    
    @staticmethod
    def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
        """Label smoothing cross entropy"""
        vocab_size = tf.shape(y_pred)[-1]
        confidence = 1.0 - smoothing
        
        # One-hot encode with smoothing
        smooth_labels = tf.one_hot(y_true, vocab_size, dtype=tf.float32)
        smooth_labels = smooth_labels * confidence + smoothing / tf.cast(vocab_size, tf.float32)
        
        return tf.nn.softmax_cross_entropy_with_logits(
            labels=smooth_labels, logits=y_pred
        )

# 개선된 손실 함수
def improved_masked_loss(y_true, y_pred, use_focal=False, use_label_smoothing=False):
    if use_focal:
        loss = ImprovedMetrics.focal_loss(y_true, y_pred)
    elif use_label_smoothing:
        loss = ImprovedMetrics.label_smoothing_loss(y_true, y_pred)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
    
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    masked_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    return masked_loss

def improved_masked_accuracy(y_true, y_pred):
    preds = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, preds), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / (tf.reduce_sum(mask) + 1e-8)

def improved_masked_perplexity(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_pred
    )
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    avg_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    return tf.exp(tf.minimum(avg_loss, 10.0))

def improved_top5_accuracy(y_true, y_pred):
    top5_preds = tf.nn.top_k(y_pred, k=5).indices
    top5_preds = tf.cast(top5_preds, dtype=y_true.dtype)
    y_true_expanded = tf.expand_dims(y_true, axis=-1)
    matches = tf.reduce_any(tf.equal(y_true_expanded, top5_preds), axis=-1)
    matches = tf.cast(matches, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, pad_id), tf.float32)
    return tf.reduce_sum(matches * mask) / (tf.reduce_sum(mask) + 1e-8)


# ======================= 개선된 학습률 스케줄러 ======================
def create_cosine_schedule(initial_lr=1e-4, min_lr=1e-6, warmup_steps=2000, total_steps=50000):
    """Cosine annealing with warmup"""
    def schedule(step):
        step = tf.cast(step, tf.float32)
        
        if step < warmup_steps:
            # Linear warmup
            return initial_lr * step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = tf.minimum(progress, 1.0)
            cosine_decay = 0.5 * (1 + tf.cos(math.pi * progress))
            return min_lr + (initial_lr - min_lr) * cosine_decay
    
    return schedule


# ======================= 데이터셋 생성 ======================
print("🔄 데이터셋 생성 중...")
dataset = create_efficient_dataset(train_sentences, batch_size, max_len)

# ======================= 모델 생성 및 설정 ======================
print("🏗️ 모델 생성 중...")
model = ImprovedCobraModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    dropout_rate=0.1,
    use_rms_norm=True
)

# 개선된 옵티마이저 설정
total_steps = (len(train_sentences) // batch_size) * 3  # 3 epochs
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=create_cosine_schedule(
        initial_lr=1e-4,
        min_lr=1e-6,
        warmup_steps=2000,
        total_steps=total_steps
    ),
    beta_1=0.9,
    beta_2=0.95,
    epsilon=1e-8,
    weight_decay=0.01,
    clipnorm=1.0
)

# 모델 컴파일
model.compile(
    optimizer=optimizer,
    loss=improved_masked_loss,
    metrics=[
        improved_masked_accuracy,
        improved_masked_perplexity,
        improved_top5_accuracy
    ]
)

# 더미 인풋으로 모델 초기화
dummy_input = tf.zeros((1, max_len), dtype=tf.int32)
model(dummy_input)
model.summary()

# ======================= 콜백 설정 ======================
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# ======================= 학습 시작 ======================
print("🚀 학습 시작!")
history = model.fit(
    dataset,
    epochs=3,
    steps_per_epoch=len(train_sentences) // batch_size,
    callbacks=callbacks,
    verbose=1
)

# ======================= 모델 저장 ======================
model.save_weights("ImprovedCobra.weights.h5")
print("✅ 개선된 모델 가중치 저장 완료!")

# Colab 다운로드 (필요시)
try:
    from google.colab import files
    files.download('ImprovedCobra.weights.h5')
except:
    print("Colab 환경이 아닙니다. 로컬에 저장되었습니다.")


# ======================= 개선된 생성 함수 ======================
def advanced_generate_text(
    model, 
    prompt, 
    max_len=256, 
    max_gen=200, 
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    min_len=20
):
    """고급 텍스트 생성 함수"""
    
    model_input = text_to_ids(f"<start> {prompt} <sep>")
    model_input = model_input[:max_len]
    generated = list(model_input)
    
    # 반복 방지를 위한 n-gram 추적
    seen_ngrams = set()
    ngram_size = 4
    
    for step in range(max_gen):
        # 입력 준비
        if len(generated) > max_len:
            input_seq = generated[-max_len:]
        else:
            input_seq = generated
            
        input_padded = np.pad(
            input_seq, 
            (0, max_len - len(input_seq)), 
            constant_values=pad_id
        )
        input_tensor = tf.convert_to_tensor([input_padded])
        
        # 예측
        logits = model(input_tensor, training=False)
        next_token_logits = logits[0, len(input_seq) - 1].numpy()
        
        # 반복 페널티 적용
        for i, token_id in enumerate(generated[-50:]):  # 최근 50개 토큰에 페널티
            if token_id < len(next_token_logits):
                next_token_logits[token_id] /= repetition_penalty
        
        # 특수 토큰 조정
        next_token_logits[pad_id] -= 10.0
        if len(generated) < min_len:
            next_token_logits[end_id] -= 5.0
            
        # 온도 적용
        next_token_logits = next_token_logits / temperature
        
        # Top-k 필터링
        if top_k > 0:
            top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
            top_k_logits = next_token_logits[top_k_indices]
            filtered_logits = np.full_like(next_token_logits, -np.inf)
            filtered_logits[top_k_indices] = top_k_logits
            next_token_logits = filtered_logits
        
        # Top-p 필터링
        probs = tf.nn.softmax(next_token_logits).numpy()
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        cutoff = np.searchsorted(cumulative_probs, top_p)
        top_indices = sorted_indices[:cutoff + 1]
        top_probs = sorted_probs[:cutoff + 1]
        top_probs /= np.sum(top_probs)
        
        # 토큰 선택
        next_token_id = np.random.choice(top_indices, p=top_probs)
        
        # N-gram 반복 체크
        if len(generated) >= ngram_size:
            current_ngram = tuple(generated[-(ngram_size-1):] + [next_token_id])
            if current_ngram in seen_ngrams:
                # 반복되는 n-gram이면 다른 토큰 선택
                remaining_indices = [idx for idx in top_indices if idx != next_token_id]
                if remaining_indices:
                    remaining_probs = probs[remaining_indices]
                    remaining_probs /= np.sum(remaining_probs)
                    next_token_id = np.random.choice(remaining_indices, p=remaining_probs)
            seen_ngrams.add(current_ngram)
        
        # 종료 조건
        if next_token_id == end_id and len(generated) >= min_len:
            break
            
        generated.append(int(next_token_id))
    
    return ids_to_text(generated)


# ======================= 테스트 ======================
print("\n\n===== 개선된 모델 생성 결과 =====")
test_prompts = [
    "안녕하세요",
    "파이썬으로 웹 크롤링을 하려면",
    "건강한 식단을 위한 조언",
    "인공지능의 미래는"
]

for prompt in test_prompts:
    print(f"\n프롬프트: {prompt}")
    result = advanced_generate_text(
        model, 
        prompt, 
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    print(f"응답: {result}")
    print("-" * 50)
