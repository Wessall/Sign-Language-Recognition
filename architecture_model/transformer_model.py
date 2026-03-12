import tensorflow as tf
from tensorflow.keras import layers, models


def get_model(max_len, channels, num_classes, dim=64, pad_value=-100.0):

    class MultiHeadSelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim=256, num_heads=4, dropout=0.0):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.qkv = layers.Dense(3 * dim, use_bias=False)
            self.drop1 = layers.Dropout(dropout)
            self.proj = layers.Dense(dim, use_bias=False)
            self.supports_masking = True

        def call(self, inputs, mask=None, training=None):
            B = tf.shape(inputs)[0]
            T = tf.shape(inputs)[1]

            qkv = self.qkv(inputs)
            qkv = tf.reshape(qkv, (B, T, self.num_heads, 3 * self.head_dim))
            qkv = tf.transpose(qkv, perm=(0, 2, 1, 3))
            q, k, v = tf.split(qkv, 3, axis=-1)

            attn = tf.matmul(q, k, transpose_b=True) * self.scale

            if mask is not None:
                mask_expanded = tf.cast(mask, tf.float32)[:, None, None, :]
                attn = attn + (1.0 - mask_expanded) * (-1e9)

            attn = tf.nn.softmax(attn, axis=-1)
            attn = self.drop1(attn, training=training)

            x = tf.matmul(attn, v)
            x = tf.transpose(x, perm=(0, 2, 1, 3))
            x = tf.reshape(x, (B, T, self.dim))
            return self.proj(x)

    def TransformerBlock(dim=64, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1):
        def apply(inputs):
            x = layers.BatchNormalization(momentum=0.95)(inputs)
            x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            x = layers.Add()([inputs, x])
            attn_out = x

            x = layers.BatchNormalization(momentum=0.95)(x)
            x = layers.Dense(dim * expand, activation="swish", use_bias=False)(x)
            x = layers.Dense(dim, use_bias=False)(x)
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            x = layers.Add()([attn_out, x])
            return x
        return apply

    def Conv1DBlock(dim, ksize, drop_rate=0.1):
        def apply(x):
            shortcut = x
            x = layers.BatchNormalization(momentum=0.95)(x)
            x = layers.Conv1D(dim, ksize, padding="same", activation="swish", use_bias=False)(x)
            x = layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
            x = layers.Add()([shortcut, x])
            return x
        return apply

    inp = layers.Input(shape=(max_len, channels), name="input_features")
    x = layers.Masking(mask_value=pad_value)(inp)

    x = layers.Dense(dim, use_bias=False, name="stem_dense")(x)
    x = layers.BatchNormalization(momentum=0.95, name="stem_bn")(x)

    ksize = 17

    x = Conv1DBlock(dim, ksize)(x)
    x = Conv1DBlock(dim, ksize)(x)
    x = Conv1DBlock(dim, ksize)(x)

    x = TransformerBlock(dim, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1)(x)  # head_dim = 64//4 = 16 ✓
    x = TransformerBlock(dim, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1)(x)

    x = Conv1DBlock(dim, ksize)(x)
    x = Conv1DBlock(dim, ksize)(x)
    x = Conv1DBlock(dim, ksize)(x)

    x = TransformerBlock(dim, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1)(x)  # CHANGED: 8→4, head_dim=16 ✓
    x = TransformerBlock(dim, num_heads=4, expand=2, attn_dropout=0.1, drop_rate=0.1)(x)

    x = layers.Dense(dim * 2, activation="swish", name="top_dense")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4, name="top_dropout")(x)
    x = layers.Dense(dim, activation="swish", name="pre_classifier")(x)
    outputs = layers.Dense(num_classes, name="classifier")(x)

    model = models.Model(inputs=inp, outputs=outputs, name="transformer_model")
    return model