import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(max_len, channels, num_classes, dim=256, dropout_rate=0.4):
    """
    Constructs a robust BiLSTM model for sequential feature classification.
    """
    inp = layers.Input(shape=(max_len, channels), name='input_features')

    # Masking layer to ignore the PAD value (-100.0)
    x = layers.Masking(mask_value=-100.0, input_shape=(max_len, channels), name='masking_layer')(inp)

    # Feature extraction (Dimensionality reduction)
    x = layers.Dense(dim, activation='swish', name='feature_dense')(x)  # CHANGED: relu → swish
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.Dropout(dropout_rate)(x)

    # Sequential Modeling (BiLSTM)
    x = layers.Bidirectional(layers.LSTM(dim, return_sequences=True), name='bilstm_1')(x)
    x = layers.Dropout(0.2)(x)        # CHANGED: dropout_rate → 0.2, LSTM needs less dropout

    x = layers.Bidirectional(layers.LSTM(dim // 2), name='bilstm_2')(x)
    x = layers.Dropout(0.2)(x)        # CHANGED: dropout_rate → 0.2

    # Classification Head
    x = layers.Dense(128, activation='swish', name='dense_head')(x)  # CHANGED: relu → swish
    x = layers.BatchNormalization(name='bn_2')(x)
    out = layers.Dense(num_classes, name='classifier')(x)  # CHANGED: removed softmax → from_logits=True

    model = models.Model(inputs=inp, outputs=out, name="BiLSTM_Baseline")
    return model