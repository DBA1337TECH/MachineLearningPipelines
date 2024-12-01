from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import librosa
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.layers import Input, LSTM, Dense
import urllib.request

# def download_fsdd(target_dir='./audio'):
#     os.makedirs(target_dir, exist_ok=True)
#     url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
#     urllib.request.urlretrieve(url, filename=os.path.join(target_dir, 'fsdd.zip'))
#
# download_fsdd()

def load_and_preprocess_audio_files(path, sample_rate=22050, sequence_length=1024, max_files=10):
    audio_data = []
    target_data = []

    files = glob.glob(os.path.join(path, '*.wav'))
    for file in files[:max_files]:  # Limit the number of files to load
        data, _ = librosa.load(file, sr=sample_rate)
        if len(data) < sequence_length:
            continue  # Skip files that are too short
        # Create overlapping windows of data
        for start_idx in range(0, len(data) - sequence_length, sequence_length // 2):
            end_idx = start_idx + sequence_length
            audio_data.append(data[start_idx:end_idx])
            target_data.append(data[start_idx+1:end_idx+1])  # Shifted by one sample

    return np.array(audio_data), np.array(target_data)

# Load and preprocess data
audio_path = './audio/free-spoken-digit-dataset-master/recordings'  # Update this to your actual audio directory
audio_data, target_data = load_and_preprocess_audio_files(audio_path, max_files=100)
audio_data = audio_data[..., np.newaxis]  # Add channel dimension
target_data = target_data[..., np.newaxis]  # Add channel dimension

# Split data
X_train, X_test, y_train, y_test = train_test_split(audio_data, target_data, test_size=0.2, random_state=42)




def build_encoder(input_shape, latent_dim=128):
    # Input for variable-length sequences of given input_shape
    encoder_inputs = Input(shape=input_shape)
    # LSTM layer(s)
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    # The encoder model
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model

def build_decoder(input_shape, latent_dim=128):
    # Inputs for the initial states from the encoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # The input to the decoder at each time step (should match the encoder's output shape)
    decoder_inputs = Input(shape=(None, input_shape[-1]))  # None for variable sequence length
    # LSTM layer(s)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # A dense layer to convert LSTM outputs to final output (match the input feature size)
    decoder_dense = Dense(input_shape[-1], activation='linear')  # Match the feature dimension of input
    decoder_outputs = decoder_dense(decoder_outputs)

    # The decoder model
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    return decoder_model

def create_autoencoder(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Model creation
input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_channels)
autoencoder = create_autoencoder(input_shape)
decoder = None
encoder = None
def build_autoencoder(input_dim, output_dim, latent_dim=128):
    # Encoder
    encoder_inputs = Input(shape=(None, input_dim))
    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, output_dim))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='tanh')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Full model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model

def build_seq2seq_model(input_shape, latent_dim=128, output_steps=100, output_features=40):
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
    encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, output_features), name='decoder_inputs')  # Variable length
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_features, activation='linear', name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# Assuming the number of timesteps and features in your input data
input_shape = (None, 40)  # None for variable sequence length, 40 features per timestep
seq2seq_model = build_seq2seq_model(input_shape, output_features=40)


import numpy as np

# Example data generation (replace this with your actual dataset loading)
num_samples = 1000
max_sequence_length = 100
features_dim = 40

# Randomly generate some data
encoder_input_data = np.random.random((num_samples, max_sequence_length, features_dim))
decoder_input_data = np.zeros_like(encoder_input_data)  # assuming a start token of zeros
decoder_target_data = np.roll(encoder_input_data, -1, axis=1)  # shift everything to the left

# Training configuration
seq2seq_model.compile(optimizer='adam', loss='mse')
seq2seq_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=64,
                  epochs=50,
                  validation_split=0.2)


# Model configuration
input_dim = 1  # Example input dimension, modify as per your data
output_dim = 1  # Example output dimension
input_shape = (X_train.shape[1], X_train.shape[2])
# Build models
autoencoder = create_autoencoder(input_shape)
encoder_model = build_encoder(input_shape)
decoder_model = build_decoder(input_shape)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
# Training here - ensure you provide the appropriate input and target data
# autoencoder.fit([input_audio_data, target_audio_data], target_audio_data, epochs=20)
# Model training
# history = autoencoder.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model
# test_loss = autoencoder.evaluate(X_test, y_test)
# print('Test Loss:', test_loss)

# After training, you can save encoder and decoder models
# encoder_model.save('encoder_model')
# decoder_model.save('decoder_model')
#autoencoder.save("autoencoder_model")
# Assuming 'encoder_model' and 'decoder_model' have been trained and loaded
# encoder_model = tf.keras.models.load_model('encoder_model')
# decoder_model = tf.keras.models.load_model('decoder_model')

# Convert the encoder
# encoder_converter = tf.lite.TFLiteConverter.from_keras_model(encoder_model)
# encoder_tflite_model = encoder_converter.convert()

# Save the converted model
# with open('encoder_model.tflite', 'wb') as f:
#     f.write(encoder_tflite_model)

# Convert the decoder
# decoder_converter = tf.lite.TFLiteConverter.from_keras_model(decoder_model)
# decoder_tflite_model = decoder_converter.convert()

# Save the converted model
# with open('decoder_model.tflite', 'wb') as f:
#     f.write(decoder_tflite_model)


def load_tflite_model(tflite_model_path):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, input_data):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return output_data


# Load the TFLite models
# encoder_interpreter = load_tflite_model('encoder_model.tflite')
# decoder_interpreter = load_tflite_model('decoder_model.tflite')
autoencoder_model = tf.keras.models.load_model("saved_models")
# Prepare some input data, e.g., a batch of sequences
# input_data should be shaped correctly for the model's input
# Here we assume it is prepared appropriately

from tensorflow.keras.models import Model

# Assuming 'seq2seq_model' is the trained model

# Extract the encoder
encoder_inputs = seq2seq_model.input[0]  # this is the input for the encoder
encoder_outputs, state_h_enc, state_c_enc = seq2seq_model.layers[2].output  # LSTM layer outputs
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('encoder_model_extracted')

# Extract the decoder
decoder_inputs = seq2seq_model.input[1]  # input for the decoder
decoder_state_input_h = Input(shape=(128,), name='input_h')  # latent_dim is 128 as defined
decoder_state_input_c = Input(shape=(128,), name='input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = seq2seq_model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = seq2seq_model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)
decoder_model.save('decoder_model_extracted')


from tensorflow.keras.models import load_model

# Load the models back from disk
loaded_encoder_model = load_model('encoder_model_extracted')
loaded_decoder_model = load_model('decoder_model_extracted')

# Use the loaded models for inference or further processing as required


input_data = np.random.random((1, 1, 40)).astype(np.float32)  # Example data

# Run inference with the encoder
encoder_states = loaded_encoder_model(input_data)
# output = autoencoder_model(input_data)
print(encoder_states)

# Assume the decoder expects the initial states from the encoder
# and you have some sequence for the decoder to work on (decoder_input_data)
# You need to set up the decoder's input correctly, which may include setting initial states
# decoder_input_data = np.random.random((1, 1024, 1)).astype(np.float32)  # Example data
# decoder_output = run_inference(decoder_model, encoder_states)
decoder_output = loaded_decoder_model(encoder_states)
print("Decoder output shape:", decoder_output.shape)

