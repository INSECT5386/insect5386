class RecurrentFFN(tf.keras.layers.Layer):
    """SwiGLU + tanh를 사용한 재귀형 Feed-Forward Network"""
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = hidden_dim or int(input_dim * 8 / 3)  # SwiGLU 표준 비율

        # Hidden State 초기화 (학습 가능한 변수)
        self.hidden_state = tf.Variable(
            tf.zeros((1, input_dim)), trainable=True, name="hidden_state"
        )

        # Dense 레이어 정의
        self.gate_proj = layers.Dense(hidden_dim, use_bias=False)
        self.up_proj = layers.Dense(hidden_dim, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)

    def reset_hidden_state(self, batch_size):
        """Batch 크기에 맞춰 Hidden State 초기화"""
        self.hidden_state.assign(tf.zeros((batch_size, self.input_dim)))

    def call(self, x, training=False):
        # Hidden State와 입력 결합
        combined = tf.concat([x, self.hidden_state], axis=-1)

        # SwiGLU 적용
        gate = self.gate_proj(combined)
        up = self.up_proj(combined)
        swiglu_output = tf.nn.silu(gate) * up  # SwiGLU 출력

        # tanh 적용 및 최종 Hidden State 계산
        tanh_output = tf.tanh(combined)
        new_hidden_state = swiglu_output * tanh_output

        # Dropout 적용
        new_hidden_state = self.dropout(new_hidden_state, training=training)

        # Hidden State 갱신
        self.hidden_state.assign(new_hidden_state)

        return new_hidden_state

class HybridBlock(layers.Layer):
    def __init__(self, d_model, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv1D(d_model, kernel_size, padding='causal')
        self.ffn = RecurrentFFN(d_model)  # 재귀형 FFN 사용
        self.norm = layers.LayerNormalization()

    def call(self, x):
        x = self.conv(x)
        x = self.ffn(x)  # 재귀형 FFN 적용
        x = self.norm(x)
        return x

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
