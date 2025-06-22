class RecurrentDecoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim=None, num_layers=2, dropout_rate=0.1):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.layers = [RecurrentFFN(embed_dim, hidden_dim, dropout_rate) for _ in range(num_layers)]
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden_states, training=False):
        x = self.embedding(x)
        new_hidden_states = []

        for i, layer in enumerate(self.layers):
            x, new_state = layer(x, hidden_states[i], training=training)
            new_hidden_states.append(new_state)

        x = self.norm(x)
        logits = self.head(x)
        return logits, new_hidden_states

    def get_initial_states(self, batch_size, dtype=tf.float32):
        return [layer.get_initial_state(batch_size=batch_size, dtype=dtype) for layer in self.layers]

# 인코더
encoder_input = Input(shape=(max_len_q,))
encoder_emb = Embedding(vocab_size, 50)(encoder_input)

encoder = RFNN(50, return_sequences=True, return_state=True, name='encoder_1')
encoder_output, state_h, state_c = encoder_1(encoder_emb)

# 디코더
decoder_input = Input(shape=(max_len_a,))
decoder_emb = Embedding(vocab_size_a, 50)(decoder_input)

# 첫 번째 LSTM (초기 상태는 encoder에서 나오는 상태 사용)
decoder_1 = RFFN(50, return_sequences=True, return_state=True, name='decoder_lstm_1',
                      kernel_initializer=initializers.GlorotUniform(), recurrent_initializer=initializers.Orthogonal())
decoder_output, state_h_1, state_c_1 = decoder_1(decoder_emb, initial_state=[state_h, state_c])

decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_output)

# 모델 정의
model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_outputs)
