from __future__ import absolute_import, division, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import numpy as np
import os
import time
import discord

checkpoint_dir = './training_checkpoints'
  
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_prefix,
      save_weights_only=True)

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(
    tf.keras.layers.GRU, recurrent_activation='sigmoid')

batch_size = 64

buffer_size = 10000

embedding_dim = 256

rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
        batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        stateful=True),
    rnn(rnn_units,
        return_sequences=True,
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def create_model(data):

  model = build_model(data.vocab_size, embedding_dim, rnn_units, batch_size=batch_size)

  model.compile(
      optimizer = tf.compat.v1.train.AdamOptimizer(),
      loss = loss)

  return model

def load_weights(model):
  try:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
  except Exception as e:
    raise Exception("You must train first. If you're getting this error but you already have checkpoints saved in ./training_checkpoints then something is wrong. You can try deleting the checkpoints and starting again or starting an issue on the git repository.")
  return model
def load_model(data):

  model = build_model(data.vocab_size, embedding_dim, rnn_units, batch_size=batch_size)

  model.compile(
      optimizer = tf.compat.v1.train.AdamOptimizer(),
      loss = loss)

  model = load_weights(model)

  model.build(tf.TensorShape([1, None]))
  return model

def output_mode(model, data):

  model = build_model(data.vocab_size, embedding_dim, rnn_units, batch_size=1)
  model = load_weights(model)
  model.build(tf.TensorShape([1, None]))
  return model

def train_model(model, data, epochs):
  history = model.fit(data.dataset.repeat(), epochs=epochs, steps_per_epoch=data.steps_per_epoch, callbacks=[checkpoint_callback])

def generate_text(model, data, start_string):
  # Max number of characters to generate
  num_generate = 1500

  input_eval = [data.char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []

  temperature = 0.5

  model.reset_states()
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)
    char = data.idx2char[predicted_id]
    text_generated.append(data.idx2char[predicted_id])
    if (char == "\n"):
        break
  return (''.join(text_generated))

class data():
  def __init__(self, path_to_file):
    self.text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    self.vocab = sorted(set(self.text))

    self.char2idx = {u:i for i, u in enumerate(self.vocab)}
    self.idx2char = np.array(self.vocab)

    self.text_as_int = np.array([self.char2idx[c] for c in self.text])

    self.seq_length = 100

    self.examples_per_epoch = len(self.text)//self.seq_length
    self.steps_per_epoch = self.examples_per_epoch//batch_size

    self.char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)

    self.sequences = self.char_dataset.batch(self.seq_length+1, drop_remainder=True)

    self.vocab_size = len(self.vocab)
    print(self.vocab_size)

    self.dataset = self.sequences.map(self.split_input_target)
    self.dataset = self.dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
  
  def split_input_target(self,chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text