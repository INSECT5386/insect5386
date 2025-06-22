import tensorflow as tf


class RecurrentFFN(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=None, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim * 4

        # Update Gate and Reset Gate (like GRU)
        self.update_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.Activation('sigmoid')
        ])
        self.reset_gate = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim),
            tf.keras.layers.Activation('sigmoid')
        ])

        # Candidate hidden state (SwiGLU 기반)
        self.gate_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)
        self.up_proj = tf.keras.layers.Dense(self.hidden_dim, use_bias=True)

        # Down projection
        self.down_proj = tf.keras.layers.Dense(input_dim, use_bias=True)

        # Layer Normalizations
        self.norm_hidden = tf.keras.layers.LayerNormalization()
        self.norm_output = tf.keras.layers.LayerNormalization()

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, hidden_state, training=False):
        # Update Gate & Reset Gate 계산
        combined = tf.concat([x, hidden_state], axis=-1)
        update_gate = self.update_gate(combined)
        reset_gate = self.reset_gate(combined)

        # Reset 적용 후 candidate 생성
        gated_hidden = reset_gate * hidden_state
        candidate_combined = tf.concat([x, gated_hidden], axis=-1)

        # SwiGLU 활성화
        gate = self.gate_proj(candidate_combined)
        up = self.up_proj(candidate_combined)
        swiglu_output = tf.nn.silu(gate) * up

        # Hidden State 업데이트
        new_hidden_state = (1 - update_gate) * hidden_state + update_gate * swiglu_output
        new_hidden_state = self.norm_hidden(new_hidden_state)

        # Final output
        output = self.down_proj(new_hidden_state)
        output = self.norm_output(output)
        output = self.dropout(output, training=training)

        return output, [new_hidden_state]

    def get_initial_state(self, batch_size=None, dtype=None):
        """Returns initial hidden state"""
        return tf.zeros(shape=[batch_size, self.hidden_dim], dtype=dtype)

# 인코더
encoder_input = Input(shape=(max_len_q,))
encoder_emb = Embedding(vocab_size, 50)(encoder_input)
rnn_cell = RecurrentFFN(hidden_units)

encoder = RNN(rnn_cell, return_sequences=True, return_state=True, name='encoder_1')
encoder_output, new_hidden_state = encoder_1(encoder_emb)

# 디코더
decoder_input = Input(shape=(max_len_a,))
decoder_emb = Embedding(vocab_size_a, 50)(decoder_input)

rnn_cell_2 = RecurrentFFN(hidden_units)

# 첫 번째 LSTM (초기 상태는 encoder에서 나오는 상태 사용)
decoder_1 = RNN(rnn_cell_2, return_sequences=True, return_state=True, name='decoder_1',
                      kernel_initializer=initializers.GlorotUniform(), recurrent_initializer=initializers.Orthogonal())
decoder_output, new_hidden_state = decoder_1(decoder_emb, initial_state=[new_hidden_state])

decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_output)

# 모델 정의
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
