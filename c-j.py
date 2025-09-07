class RotaryPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        self.inv_freq = tf.constant(inv_freq, dtype=tf.float32)

    def call(self, x):
        b, h, s, d = tf.unstack(tf.shape(x))
        t = tf.range(s, dtype=tf.float32)
        freqs = tf.einsum('i,j->ij', t, self.inv_freq)
        emb_sin = tf.sin(freqs)
        emb_cos = tf.cos(freqs)
        emb_cos = tf.reshape(emb_cos, [1, 1, s, -1])
        emb_sin = tf.reshape(emb_sin, [1, 1, s, -1])
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = tf.stack([x1 * emb_cos - x2 * emb_sin, x1 * emb_sin + x2 * emb_cos], axis=-1)
        x_rot = tf.reshape(x_rot, tf.shape(x))
        return x_rot

class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(d_ff, dtype=tf.float32)
        self.out = tf.keras.layers.Dense(d_model, dtype=tf.float32)

    def call(self, x):
        x_proj = self.proj(x)
        x_val, x_gate = tf.split(x_proj, 2, axis=-1)
        return self.out(x_val * tf.nn.silu(x_gate))

class Adapter(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.proj = tf.keras.layers.Dense(64, dtype=tf.float32)
        self.out = tf.keras.layers.Dense(d_model, dtype=tf.float32)

    def call(self, x):
        x_proj = self.proj(x)
        x = tf.nn.gelu(x_proj)
        return self.out(x)

class GPTBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=8, dropout_rate=0.1, adapter_dim=64):
        super().__init__()
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype=tf.float32)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dtype=tf.float32)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.adapter_down = tf.keras.layers.Dense(adapter_dim, activation='gelu', dtype=tf.float32)
        self.adapter_up = tf.keras.layers.Dense(d_model, dtype=tf.float32)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype=tf.float32)
        self.ffn = SwiGLU(d_model, d_ff)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.rope = RotaryPositionalEmbedding(d_model // num_heads)

    def call(self, x, training=False):
        x_norm = self.ln1(x)
        b, s, d = tf.shape(x_norm)[0], tf.shape(x_norm)[1], x_norm.shape[-1]
        h = self.mha.num_heads
        dh = d // h
        qkv = tf.reshape(x_norm, [b, s, h, dh])
        qkv = tf.transpose(qkv, [0, 2, 1, 3])
        q = self.rope(qkv)
        k = self.rope(qkv)
        q = tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [b, s, h * dh])
        k = tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [b, s, h * dh])
        attn_out = self.mha(query=q, value=x_norm, key=k, use_causal_mask=True, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        adapter_out = self.adapter_up(self.adapter_down(attn_out))
        attn_out = attn_out + adapter_out
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout2(ffn_out, training=training)
        return x

class InLaM(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, d_model, d_ff, n_layers, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, dtype=tf.float32)
        self.blocks = [GPTBlock(d_model, d_ff, num_heads, dropout_rate) for _ in range(n_layers)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5, dtype=tf.float32)
        self.lm_head = tf.keras.layers.Dense(vocab_size, use_bias=False, dtype=tf.float32)

    def call(self, x, training=False):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits  # 이미 float32
