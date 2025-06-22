import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, RNN
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform, Orthogonal


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

    def build(self, input_shape):
        pass

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

        return output, new_hidden_state

    def get_initial_state(self, batch_size=None, dtype=None):
        """Returns initial hidden state"""
        return tf.zeros(shape=[batch_size, self.hidden_dim], dtype=dtype)


# 가정된 파라미터
vocab_size = 10000
max_len = 256

# 인코더
encoder_input = tf.keras.Input(shape=(max_len,))
encoder_emb = tf.keras.layers.Embedding(vocab_size, 50)(encoder_input)

rnn_cell = RecurrentFFN(input_dim=50, hidden_dim=200)
encoder = tf.keras.layers.RNN(rnn_cell, return_sequences=True, return_state=True, name='encoder')
encoder_output, encoder_final_state = encoder(encoder_emb)

# 디코더
decoder_input = tf.keras.Input(shape=(max_len,))
decoder_emb = tf.keras.layers.Embedding(vocab_size, 50)(decoder_input)

rnn_cell_decoder = RecurrentFFN(input_dim=50, hidden_dim=200)

decoder = tf.keras.layers.RNN(
    rnn_cell_decoder,
    return_sequences=True,
    return_state=True,
    name='decoder',
)

decoder_output, _ = decoder(decoder_emb, initial_state=encoder_final_state)

# 출력층
decoder_dense = tf.keras.layers.TimeDistributed(
    tf.keras.layers.Dense(vocab_size, activation='softmax')
)
decoder_outputs = decoder_dense(decoder_output)

# 모델 정의
model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
