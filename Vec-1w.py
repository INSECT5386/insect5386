class RecurrentFFN(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        hidden_dim = hidden_dim or int(input_dim * 8 / 3)

        # Dense Layers
        self.gate_proj = tf.keras.layers.Dense(hidden_dim, use_bias=True)
        self.up_proj = tf.keras.layers.Dense(hidden_dim, use_bias=True)
        self.down_proj = tf.keras.layers.Dense(input_dim, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, hidden_state, training=False):
        combined = tf.concat([x, hidden_state], axis=-1)

        # SwiGLU
        gate = self.gate_proj(combined)
        up = self.up_proj(combined)
        swiglu_output = tf.nn.silu(gate) * up

        # tanh 적용
        tanh_output = tf.tanh(combined)
        new_hidden_state = swiglu_output * tanh_output
        new_hidden_state = self.dropout(new_hidden_state, training=training)

        # 최종 출력
        output = self.down_proj(new_hidden_state)

        return output, new_hidden_state
