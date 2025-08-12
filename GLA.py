class GLALayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"

        self.qkv = tf.keras.layers.Dense(dim * 3)
        self.out_proj = tf.keras.layers.Dense(dim)

    def call(self, inputs, training=None, mask=None):
        B, T, D = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        qkv = self.qkv(inputs)
        q, k, v = tf.split(qkv, 3, axis=-1)

    # Split heads
        q = tf.reshape(q, (B, T, self.num_heads, self.head_dim))
        k = tf.reshape(k, (B, T, self.num_heads, self.head_dim))
        v = tf.reshape(v, (B, T, self.num_heads, self.head_dim))

    # Transpose for multi-head attention
        q = tf.transpose(q, [0, 2, 1, 3])  # [B, H, T, hd]
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

    # Global latent attention
        k = tf.nn.softmax(k, axis=-1)
        context = tf.matmul(k, v, transpose_a=True)  # [B, H, hd, hd]

        if mask is not None:
            mask = mask[tf.newaxis, tf.newaxis, :, :]  # -> (1, 1, T, T)
            context = context * mask  # Broadcasting으로 자동 확장됨
    
        out = tf.matmul(q, context)  # [B, H, T, hd]
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, T, H, hd]
        out = tf.reshape(out, (B, T, D))  # [B, T, D]
    
        return self.out_proj(out)
