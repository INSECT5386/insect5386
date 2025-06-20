import tensorflow as tf
from tensorflow.keras.layers import Embedding, RNN, Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

class VecAwCell(Layer):
    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(VecAwCell, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.state_size = [tf.TensorShape(units), tf.TensorShape(units)]  # [shortterm, longterm]

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # FRU 가중치
        self.kernel_x = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='he_normal',
            name='kernel_x'
        )
        self.kernel_h = self.add_weight(
            shape=(self.units, self.units),
            initializer='he_normal',
            name='kernel_h'
        )

        # 멀티헤드 어텐션 파라미터
        self.num_heads = 8
        self.head_dim = self.units // self.num_heads

        self.W_q = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_q'
        )
        self.W_k = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_k'
        )
        self.W_v = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_v'
        )
        self.W_o = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='W_o'
        )

        # 게이트 메커니즘
        self.forget_gate = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='forget_gate'
        )
        self.input_gate = self.add_weight(
            shape=(self.units, self.units),
            initializer='glorot_normal',
            name='input_gate'
        )

        # 정규화 & 드롭아웃
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        self.dropout = Dropout(self.dropout_rate)

        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        else:
            dtype = tf.float32

        return [
            tf.zeros((batch_size, self.units), dtype=dtype),
            tf.zeros((batch_size, self.units), dtype=dtype)
        ]

    def multi_head_attention(self, query, key, value):
        """멀티헤드 어텐션 구현"""
        batch_size = tf.shape(query)[0]

        q = tf.matmul(query, self.W_q)
        k = tf.matmul(key, self.W_k)
        v = tf.matmul(value, self.W_v)

        # 분할: (batch_size, num_heads, head_dim)
        q = tf.reshape(q, (batch_size, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, self.num_heads, self.head_dim))

        # 스코어 계산
        dk = tf.cast(self.head_dim, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scores, axis=-1)

        # 가중합
        attended = tf.matmul(weights, v)

        # 다시 합침
        attended = tf.reshape(attended, (batch_size, self.units))

        # 최종 선형 변환
        output = tf.matmul(attended, self.W_o)

        return output

    def call(self, inputs, states, training=None):
        h_prev, longterm_prev = states

        x_t_proj = tf.matmul(inputs, self.kernel_x)
        h_prev_proj = tf.matmul(h_prev, self.kernel_h)
        x = x_t_proj + h_prev_proj

        # 게이트 계산
        forget_gate = tf.sigmoid(tf.matmul(x, self.forget_gate))
        input_gate = tf.sigmoid(tf.matmul(x, self.input_gate))

        # LSTM 스타일의 셀 상태 업데이트
        candidate = tf.tanh(x)
        longterm = forget_gate * longterm_prev + input_gate * candidate

        mediumterm = tf.nn.swish(x)
        shortterm = tf.nn.swish(longterm + mediumterm)

        # 정규화
        shortterm = self.layer_norm1(shortterm)
        longterm = self.layer_norm2(longterm)

        # 멀티헤드 어텐션 적용 (Self-Attention)
        attended = self.multi_head_attention(shortterm, shortterm, shortterm)

        # 잔차 연결
        output = shortterm + attended

        # 드롭아웃
        if training:
            output = self.dropout(output, training=training)

        return output, [output, longterm]


# 공유 임베딩 레이어
shared_embedding = Embedding(
    input_dim=48000,
    output_dim=256,
    mask_zero=True,
    name='shared_embedding'
)
# 🔥 명시적으로 build() 호출
shared_embedding.build(input_shape=(None,))  # (batch_size, seq_len)


class VecAwEncoder(Model):
    def __init__(self, shared_embedding, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = shared_embedding
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

    def call(self, inputs, training=None):
        mask = self.embedding.compute_mask(inputs)
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        outputs, h_state, longterm_state = self.rnn(x, mask=mask, training=training)
        return outputs, [h_state, longterm_state]


class VecAwDecoder(Model):
    def __init__(self, shared_embedding, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = shared_embedding
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

        # 출력층 - 임베딩과 가중치 공유
        vocab_size = int(shared_embedding.input_dim)
        self.output_layer = Dense(vocab_size, use_bias=False, name='output_layer')
        self.output_layer.build((None, hidden_units))
        self.output_layer.set_weights([tf.transpose(shared_embedding.weights[0])])
    def call(self, inputs, initial_state, training=None):
        mask = self.embedding.compute_mask(inputs)
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)

        rnn_out, h_state, longterm_state = self.rnn(x, initial_state=initial_state, mask=mask, training=training)
        logits = self.output_layer(rnn_out)

        return logits, [h_state, longterm_state]


class VecAwSeq2Seq(Model):
    def __init__(self, shared_embedding, hidden_units,
                 start_token_id=1, end_token_id=2, max_length=50,
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = VecAwEncoder(shared_embedding, hidden_units, dropout_rate)
        self.decoder = VecAwDecoder(shared_embedding, hidden_units, dropout_rate)

        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_length = max_length

    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            encoder_input, decoder_input = inputs
        else:
            return self.predict_step(inputs)

        encoder_output, encoder_state = self.encoder(encoder_input, training=training)
        decoder_output, _ = self.decoder(decoder_input, initial_state=encoder_state, training=training)

        return decoder_output

    def predict_step(self, encoder_input):
        batch_size = tf.shape(encoder_input)[0]

        # 인코더 실행
        encoder_output, encoder_state = self.encoder(encoder_input, training=False)

        # 디코더 시작 토큰
        decoder_input = tf.fill((batch_size, 1), self.start_token_id)

        # 결과 저장용
        outputs = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        for i in tf.range(self.max_length):
            logits, next_state = self.decoder(decoder_input, initial_state=encoder_state, training=False)
            pred_ids = tf.argmax(logits[:, -1:], axis=-1, output_type=tf.int64)

            outputs = outputs.write(i, tf.squeeze(pred_ids, axis=1))
            decoder_input = pred_ids
            encoder_state = next_state

            # 종료 토큰 체크
            if tf.reduce_all(tf.equal(pred_ids, self.end_token_id)):
                break

        outputs = outputs.stack()
        outputs = tf.transpose(outputs, perm=[1, 0])
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'start_token_id': self.start_token_id,
            'end_token_id': self.end_token_id,
            'max_length': self.max_length
        })
        return config

model = VecAwSeq2Seq(shared_embedding, hidden_units=256)

# 학습 시
encoder_inputs = tf.constant([[1, 2, 3, 4], [1, 5, 6, 7]])
decoder_inputs = tf.constant([[1, 8, 9], [1, 10, 11]])
targets = tf.constant([[8, 9, 2], [10, 11, 2]])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit([encoder_inputs, decoder_inputs], targets, epochs=10)
model.summary()
