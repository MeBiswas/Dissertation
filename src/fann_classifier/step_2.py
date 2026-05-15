# src/fann_classifier/step_2.py

from tensorflow import keras
from tensorflow.keras import layers

# ═════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═════════════════════════════════════════════════════════════════════════════
def build_fann(
    input_dim    : int   = 21,
    hidden_units : int   = 42,
    learning_rate: float = 0.001,
) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(
            hidden_units,
            activation='tanh',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            kernel_initializer='glorot_uniform',
            name='hidden',
        ),
        layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name='output',
        ),
    ], name='FANN_STBIA')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model