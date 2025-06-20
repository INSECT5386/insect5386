from tensorflow.keras.layers import Embedding, RNN
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import Layers

class VecAwCell(Layer):
    def __init__(self, units, **kwargs):
        super(VecAwCell, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # FRU 가중치
        self.kernel_x = self.add_weight(shape=(feature_dim, self.units),
                                        initializer='he_normal',
                                        name='kernel_x')
        self.kernel_h = self.add_weight(shape=(self.units, self.units),
                                        initializer='he_normal',
                                        name='kernel_h')

        # 어텐션용 가중치
        self.W_q = self.add_weight(shape=(self.units, self.units),
                                    initializer='glorot_normal',
                                    name='W_q')
        self.W_k = self.add_weight(shape=(self.units, self.units),
                                    initializer='glorot_normal',
                                    name='W_k')
        self.W_v = self.add_weight(shape=(self.units, self.units),
                                    initializer='glorot_normal',
                                    name='W_v')

        self.built = True

    def call(self, inputs, states):
        h_prev, longterm_prev = states

        x_t_proj = tf.matmul(inputs, self.kernel_x)
        h_prev_proj = tf.matmul(h_prev, self.kernel_h)
        x = x_t_proj + h_prev_proj

        longterm = tf.sigmoid(x) * tf.nn.gelu(x) + longterm_prev
        mediumterm = tf.nn.swish(x)
        shortterm = tf.nn.swish(longterm + mediumterm)

        # 내부 어텐션
        q = tf.matmul(shortterm, self.W_q)
        k = tf.matmul(longterm, self.W_k)
        v = tf.matmul(mediumterm, self.W_v)

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.units, tf.float32))
        weights = tf.nn.softmax(scores)
        attended_value = tf.matmul(weights, v)

        output = tf.nn.swish(attended_value + shortterm)

        return output, [output, longterm]

class VecAwEncoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn_cell = VecAwCell(hidden_units)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        outputs, h_state, longterm_state = self.rnn(x)
        return outputs, [h_state, longterm_state]


class VecAwDecoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn_cell = VecAwCell(hidden_units)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        rnn_out, h_state, longterm_state = self.rnn(x, initial_state=initial_state)
        logits = self.output_layer(rnn_out)
        return logits, [h_state, longterm_state]


class VecAwSeq2Seq(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VecAwEncoder(vocab_size, embedding_dim, hidden_units)
        self.decoder = VecAwDecoder(vocab_size, embedding_dim, hidden_units)

    def call(self, inputs, training=None):
        encoder_input, decoder_input = inputs
        encoder_output, encoder_state = self.encoder(encoder_input)
        
        # 디코더 초기 상태에 인코더 최종 상태 전달
        decoder_output, _ = self.decoder(decoder_input, initial_state=encoder_state)
        
        return decoder_output

    def predict_step(self, inputs):
        # Greedy Decoding 구현 (간단한 예시)
        src_seq = inputs
        batch_size = tf.shape(src_seq)[0]

        encoder_output, encoder_state = self.encoder(src_seq)

        target_seq = tf.fill((batch_size, 1), start_token_id)  # <start> 토큰

        for _ in range(max_decoding_length):
            decoder_output, decoder_state = self.decoder(target_seq[:, -1:], initial_state=encoder_state)
            next_token_logits = decoder_output[:, -1:, :]
            next_token = tf.argmax(next_token_logits, axis=-1)
            target_seq = tf.concat([target_seq, next_token], axis=-1)
            encoder_state = decoder_state

        return target_seq


model = VecAwSeq2Seq(vocab_size=VOCAB_SIZE, embedding_dim=EMB_DIM, hidden_units=HIDDEN_UNITS)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
