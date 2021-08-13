import os
import numpy as np
import tensorflow as tf

#You can change the path if you want to generate a text based on something else than Shakespearean poems
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

char2idx = {unique:idx for idx, unique in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int  = np.array([char2idx[char] for char in text])

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


sequences = char_dataset.batch(seq_length+1,drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text,target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder = True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim,rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,return_sequences=True, stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)])
    return model

# Here is the code for constructing a model or in other words teaching AI the structure of the Eglish language 


"""
model = build_model(vocab_size=len(vocab),embedding_dim= embedding_dim,
                    rnn_units= rnn_units, batch_size=BATCH_SIZE)
"""

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits, from_logits=True)
"""
model.compile(optimizer='adam', loss = loss)
"""

checkpoint_dir = './training_chechpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'chkpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)
EPOCHS = 35

"""
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
"""


# This is the code for generating a text based on the constructed model
                    
"""                          
model = build_model(vocab_size, embedding_dim,rnn_units,batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1,None]))

model.summary()

def generate_text(model, start_string):
    num_generate = 4300
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval,0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))
print(generate_text(model, start_string='The '))
"""
