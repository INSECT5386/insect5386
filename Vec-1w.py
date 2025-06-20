import tensorflow as tf
from tensorflow.keras.layers import Embedding, RNN, Layer, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np

class VecAwCell(Layer):
    def __init__(self, units, dropout_rate=0.1, decay_factor=0.9, **kwargs):
        super(VecAwCell, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.decay_factor = decay_factor
        self.state_size = [units, units]  # [shortterm, longterm]

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

        # 어텐션용 가중치 (멀티헤드)
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

        # 정규화 레이어
        self.layer_norm1 = LayerNormalization()
        self.layer_norm2 = LayerNormalization()
        
        # 드롭아웃
        self.dropout = Dropout(self.dropout_rate)
        
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

        self.built = True

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """초기 상태 설정"""
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        
        return [
            tf.zeros((batch_size, self.units), dtype=dtype),  # shortterm
            tf.zeros((batch_size, self.units), dtype=dtype)   # longterm
        ]

    def multi_head_attention(self, query, key, value):
        """멀티헤드 어텐션 구현"""
        batch_size = tf.shape(query)[0]
        
        # Q, K, V 변환
        q = tf.matmul(query, self.W_q)
        k = tf.matmul(key, self.W_k)
        v = tf.matmul(value, self.W_v)
        
        # 멀티헤드로 분할
        q = tf.reshape(q, (batch_size, self.num_heads, self.head_dim))
        k = tf.reshape(k, (batch_size, self.num_heads, self.head_dim))
        v = tf.reshape(v, (batch_size, self.num_heads, self.head_dim))
        
        # 어텐션 스코어 계산
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        
        # 가중합
        attended = tf.matmul(weights, v)
        
        # 헤드 결합
        attended = tf.reshape(attended, (batch_size, self.units))
        
        # 출력 변환
        output = tf.matmul(attended, self.W_o)
        
        return output

    def call(self, inputs, states, training=None):
        h_prev, longterm_prev = states

        # 입력 변환
        x_t_proj = tf.matmul(inputs, self.kernel_x)
        h_prev_proj = tf.matmul(h_prev, self.kernel_h)
        x = x_t_proj + h_prev_proj
        
        # 게이트 메커니즘으로 장기 메모리 업데이트
        forget_gate = tf.sigmoid(tf.matmul(x, self.forget_gate))
        input_gate = tf.sigmoid(tf.matmul(x, self.input_gate))
        
        # 메모리 업데이트 (LSTM 스타일)
        candidate = tf.nn.tanh(x)
        longterm = forget_gate * longterm_prev + input_gate * candidate
        
        # 중기 및 단기 메모리
        mediumterm = tf.nn.swish(x)
        shortterm = tf.nn.swish(longterm + mediumterm)
        
        # 정규화
        shortterm = self.layer_norm1(shortterm)
        longterm = self.layer_norm2(longterm)

        # 멀티헤드 셀프 어텐션
        attended = self.multi_head_attention(shortterm, longterm, mediumterm)
        
        # 잔차 연결
        output = shortterm + attended
        
        # 드롭아웃 적용
        if training:
            output = self.dropout(output, training=training)

        return output, [output, longterm]


class VecAwEncoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)

    def call(self, inputs, training=None):
        # 패딩 마스크 생성
        mask = self.embedding.compute_mask(inputs)
        
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        
        outputs, h_state, longterm_state = self.rnn(x, mask=mask, training=training)
        return outputs, [h_state, longterm_state]


class VecAwDecoder(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.dropout = Dropout(dropout_rate)
        self.rnn_cell = VecAwCell(hidden_units, dropout_rate)
        self.rnn = RNN(self.rnn_cell, return_sequences=True, return_state=True)
        self.output_layer = Dense(vocab_size)
        self.output_dropout = Dropout(dropout_rate)

    def call(self, inputs, initial_state, training=None):
        mask = self.embedding.compute_mask(inputs)
        
        x = self.embedding(inputs)
        x = self.dropout(x, training=training)
        
        rnn_out, h_state, longterm_state = self.rnn(
            x, initial_state=initial_state, mask=mask, training=training
        )
        
        rnn_out = self.output_dropout(rnn_out, training=training)
        logits = self.output_layer(rnn_out)
        
        return logits, [h_state, longterm_state]


class VecAwSeq2Seq(Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units, 
                 start_token_id=1, end_token_id=2, max_length=50, 
                 dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.max_length = max_length
        
        self.encoder = VecAwEncoder(vocab_size, embedding_dim, hidden_units, dropout_rate)
        self.decoder = VecAwDecoder(vocab_size, embedding_dim, hidden_units, dropout_rate)

    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            encoder_input, decoder_input = inputs
        else:
            # 추론 시에는 인코더 입력만 제공
            return self.predict_step(inputs)
            
        encoder_output, encoder_state = self.encoder(encoder_input, training=training)
        decoder_output, _ = self.decoder(decoder_input, initial_state=encoder_state, training=training)
        
        return decoder_output

    def predict_step(self, inputs):
        """Beam Search 기반 추론 (간단한 Greedy Decoding으로 구현)"""
        src_seq = inputs
        batch_size = tf.shape(src_seq)[0]

        encoder_output, encoder_state = self.encoder(src_seq, training=False)

        # 시작 토큰으로 초기화
        target_seq = tf.fill((batch_size, 1), self.start_token_id)
        finished = tf.zeros((batch_size,), dtype=tf.bool)

        for step in range(self.max_length):
            decoder_output, decoder_state = self.decoder(
                target_seq, initial_state=encoder_state, training=False
            )
            
            # 마지막 시점의 로짓
            next_token_logits = decoder_output[:, -1:, :]
            next_token = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            
            # 종료 조건 체크
            is_end = tf.equal(next_token[:, 0], self.end_token_id)
            finished = tf.logical_or(finished, is_end)
            
            # 다음 토큰 추가
            target_seq = tf.concat([target_seq, next_token], axis=-1)
            encoder_state = decoder_state
            
            # 모든 시퀀스가 종료되면 중단
            if tf.reduce_all(finished):
                break

        return target_seq

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'start_token_id': self.start_token_id,
            'end_token_id': self.end_token_id,
            'max_length': self.max_length,
        })
        return config


# 사용 예시
def create_model(vocab_size=10000, embedding_dim=256, hidden_units=512):
    """모델 생성 및 컴파일"""
    model = VecAwSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        start_token_id=1,
        end_token_id=2,
        max_length=100,
        dropout_rate=0.1
    )
    
    # 손실 함수 (레이블 스무딩 적용)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        label_smoothing=0.1
    )
    
    # 옵티마이저 (학습률 스케줄링)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    
    return model

# 콜백 설정
def get_callbacks():
    """훈련용 콜백들"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_vecaw_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )
    ]
    return callbacks

# 모델 사용 예시
if __name__ == "__main__":
    # 모델 생성
    model = create_model()
    
    # 더미 데이터로 모델 빌드
    dummy_encoder_input = tf.random.uniform((32, 20), maxval=1000, dtype=tf.int32)
    dummy_decoder_input = tf.random.uniform((32, 15), maxval=1000, dtype=tf.int32)
    
    # 모델 빌드
    _ = model([dummy_encoder_input, dummy_decoder_input])
    
    print("Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    
    # 모델 구조 출력
    model.summary()
