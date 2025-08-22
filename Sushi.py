
class SimpleFFN(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = layers.Dense(dim)
        self.up_proj = layers.Dense(dim)
        self.down_proj = layers.Dense(dim)

    def call(self, x):
        gate = tf.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

# ==================== RealMambaCore =====================
class RealMambaCore(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.ffn = SimpleFFN(hidden_dim)  # 또는 'tanh'

        self.gate_proj = layers.Dense(hidden_dim)
        self.input_proj = layers.Dense(hidden_dim)

        self.A = self.add_weight(shape=(hidden_dim,),
                                 initializer=tf.keras.initializers.RandomNormal(mean=-0.5, stddev=0.1),
                                 trainable=True, name="A")
        self.B = self.add_weight(shape=(hidden_dim,),
                                 initializer='random_normal',
                                 trainable=True, name="B")
        self.C = self.add_weight(shape=(hidden_dim,),
                                 initializer='random_normal',
                                 trainable=True, name="C")
        self.D = self.add_weight(shape=(hidden_dim,),
                                 initializer='zeros',
                                 trainable=True, name="D")

        self.norm = layers.LayerNormalization()
        self.output_proj = layers.Dense(hidden_dim)

    def fft_convolve(self, u_t, kernel_t, T):
        pad_len = T - 1
        seq_len = T + pad_len

        fft_len_float = tf.math.ceil(tf.math.log(tf.cast(seq_len, tf.float32)) / tf.math.log(2.0))
        fft_len = tf.cast(2 ** fft_len_float, tf.int32)

        u_padded = tf.pad(u_t, [[0, 0], [0, 0], [pad_len, fft_len - seq_len]])
        K_padded = tf.pad(kernel_t, [[0, 0], [0, fft_len - T]])

        U_f = tf.signal.fft(tf.cast(tf.complex(u_padded, 0.0), tf.complex64))
        K_f = tf.signal.fft(tf.cast(tf.complex(K_padded, 0.0), tf.complex64))

        Y_f = U_f * tf.expand_dims(K_f, 0)
        y_full = tf.signal.ifft(Y_f)
        y_real = tf.math.real(y_full)[..., pad_len:pad_len + T]

        return y_real

    def call(self, x):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        D = self.hidden_dim

        gate = tf.nn.silu(self.gate_proj(x))
        x_proj = self.input_proj(x)
        u = gate * x_proj

        time_idx = tf.cast(tf.range(T), dtype=self.A.dtype)[:, None]
        A_pow = tf.pow(tf.expand_dims(self.A, 0), time_idx)
        kernel = self.B * A_pow

        u_t = tf.transpose(u, [0, 2, 1])
        kernel_t = tf.transpose(kernel, [1, 0])

        y_real = self.fft_convolve(u_t, kernel_t, T)
        y = tf.transpose(y_real, [0, 2, 1])

        y = self.C * y + self.D * u

        y = self.norm(y)
        y = self.ffn(y)
        y = self.output_proj(y)

        return y
