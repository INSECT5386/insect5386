import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
import numpy as np


# 1. SwiGLU FFN (기존 코드 그대로)
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


# 2. Causal Conv1D 정의
class CausalConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = dilation_rate * (kernel_size - 1)
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='valid',
            dilation_rate=dilation_rate
        )

    def call(self, inputs):
        # inputs shape: (batch, seq_len, channels)
        x = tf.pad(inputs, [[0, 0], [self.padding, 0], [0, 0]])  # causal padding
        return self.conv(x)

# 4. Conv + SwiGLU Block
class ConvSwiGLUBlock(layers.Layer):
    def __init__(self, d_model, kernel_size=3, expansion_factor=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        inner_dim = d_model * expansion_factor

        self.net = Sequential([
            CausalConv1D(inner_dim, kernel_size),
            SwiGLUAFFN(d_model),
            CausalConv1D(d_model, kernel_size),
            layers.Dropout(dropout_rate)
        ])

        self.norm = layers.LayerNormalization()
    
    def call(self, x, training=False):
        residual = x
        x = self.net(x, training=training)
        x = x + residual
        x = self.norm(x)
        return x


# 5. 전체 언어 모델 정의
class ConvSwiGLULanguageModel(Model):
    def __init__(self, vocab_size, d_model=512, depth=8, kernel_size=3, max_seq_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Embedding(vocab_size, d_model)
        self.pos_emb = tf.keras.initializers.RandomNormal()(shape=[1, max_seq_length, d_model])
        
        self.blocks = Sequential([
            ConvSwiGLUBlock(d_model, kernel_size) for _ in range(depth)
        ])

        self.head = layers.Dense(vocab_size)

    def call(self, x):
        B, T = tf.shape(x)[0], tf.shape(x)[1]
        x = self.token_emb(x)
        x = x + self.pos_emb[:, :T, :]
        x = self.blocks(x)
        return self.head(x)

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressive 텍스트 생성 함수
        """
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = tf.nn.top_k(logits, k=top_k)
                min_value = values[:, -1]
                logits = tf.where(logits < tf.expand_dims(min_value, -1), -float('Inf'), logits)

            probs = tf.nn.softmax(logits, axis=-1)
            next_token = tf.random.categorical(probs, 1)
            input_ids = tf.concat([input_ids, next_token], axis=1)

        return input_ids
