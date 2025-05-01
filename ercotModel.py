# ercotModel.py — Shared Transformer model with regularization

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add,
    GlobalAveragePooling1D, Embedding, RepeatVector, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape, output_dim, num_generators, embedding_dim=16):
    """
    Build a Transformer model with generator ID embedding and regularization.
    """
    # Inputs
    input_seq = Input(shape=input_shape, name="time_series_input")
    gen_id_input = Input(shape=(), dtype='int32', name="generator_id")

    # Generator embedding
    gen_embedding = Embedding(input_dim=num_generators, output_dim=embedding_dim)(gen_id_input)
    gen_embedding = RepeatVector(input_shape[0])(gen_embedding)

    # Concatenate with input sequence
    x = Concatenate()([input_seq, gen_embedding])

    # Transformer block
    x_norm = LayerNormalization()(x)
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(x_norm, x_norm)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.4)(x)
    outputs = Dense(output_dim)(x)

    model = Model(inputs=[input_seq, gen_id_input], outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
    return model

def make_predictions(model, X, generator_ids):
    return model.predict([X, generator_ids], verbose=0)
