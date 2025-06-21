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

class HybridBlock(layers.Layer):
    def __init__(self, d_model, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(d_model, kernel_size, padding='causal')
        self.recurrent = layers.GRU(d_model, return_sequences=True)
        self.ffn = SwiGLUFFN(d_model)
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.conv(x) + self.recurrent(x)
        x = self.ffn(x)
        return self.norm(x)

class FreeAttentionNLG(Model):
    def __init__(self, vocab_size, d_model=512, depth=8, kernel_size=3, max_seq_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = layers.Embedding(max_seq_length, d_model)  # Learned Positional Embedding

        self.blocks = Sequential([
            HybridBlock(d_model, kernel_size) for _ in range(depth)
        ])

        self.head = layers.Dense(vocab_size)

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        x = self.token_emb(x)
        positions = tf.range(T)
        x = x + self.pos_emb(positions)  # Add positional embedding
        x = self.blocks(x)
        return self.head(x)

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive 텍스트 생성 함수
        """
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            logits = logits[:, -1, :]

            if temperature <= 1e-5:  # Greedy decoding
                next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = tf.nn.top_k(logits, k=top_k)
                    min_value = tf.reduce_min(values, axis=-1, keepdims=True)
                    logits = tf.where(logits < min_value, -float('Inf'), logits)
                probs = tf.nn.softmax(logits, axis=-1)
                next_token = tf.random.categorical(probs, 1)

            input_ids = tf.concat([input_ids, tf.expand_dims(next_token, axis=-1)], axis=1)

        return input_ids
