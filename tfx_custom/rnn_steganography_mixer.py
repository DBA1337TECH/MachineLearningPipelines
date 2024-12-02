import librosa
import numpy as np
import tensorflow as tf

def preprocess_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    spectrogram = librosa.stft(y)
    magnitude, phase = np.abs(spectrogram), np.angle(spectrogram)
    return magnitude, phase  # Return both magnitude and phase for reconstruction


def reconstruct_audio(magnitude, phase):
    complex_spectrogram = magnitude * np.exp(1j * phase)
    return librosa.istft(complex_spectrogram)


from tensorflow.keras import layers, Model

def build_audio_encoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.GRU(128, return_sequences=True)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(input_shape[-1], activation="sigmoid")(x)
    return Model(inputs, outputs, name="audio_encoder")

def build_audio_decoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.GRU(128, return_sequences=True)(inputs)
    x = layers.Dense(input_shape[-1], activation="sigmoid")(x)
    return Model(inputs, x, name="audio_decoder")


def build_steganography_model(input_shape, hidden_message_shape):
    overt_audio_input = tf.keras.Input(shape=input_shape, name="overt_audio")
    hidden_message_input = tf.keras.Input(shape=hidden_message_shape, name="hidden_message")

    # Encoder
    encoder = build_audio_encoder(input_shape)
    encoded_message = encoder(hidden_message_input)

    # Mixer (merge overt and hidden)
    mixed_audio = layers.Add()([overt_audio_input, encoded_message])

    # Decoder
    decoder = build_audio_decoder(input_shape)
    decoded_message = decoder(mixed_audio)

    return Model(inputs=[overt_audio_input, hidden_message_input], outputs=[mixed_audio, decoded_message], name="steganography_model")



