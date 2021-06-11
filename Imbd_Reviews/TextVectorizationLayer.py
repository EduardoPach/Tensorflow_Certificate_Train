import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

dataset, info = tfds.load("imdb_reviews",
                          as_supervised=True,
                          with_info=True,
                          shuffle_files=True)

BATCH_SIZE = 64
BUFFER = 10000
train_dataset = dataset['train']
test_dataset = dataset['test']
unsupervised_dataset = dataset['unsupervised']

train_dataset = train_dataset.take(10000).cache()
test_dataset = test_dataset.take(3000).cache()

train_dataset = train_dataset.shuffle(BUFFER).batch(BATCH_SIZE).prefetch(1)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(1)

VOCAB_SIZE = 1000
emb_dim = 64
max_len = "masked"
EPOCHS = 500
Early = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text,label: text))

model = tf.keras.Sequential()
model.add(encoder)
model.add(tf.keras.layers.Embedding(VOCAB_SIZE,emb_dim,mask_zero=True))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32,return_sequences=False)))
model.add(tf.keras.layers.Dense(32,activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer=tf.keras.optimizers.SGD(lr=1e-4,momentum=0.9)
              ,loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=["accuracy"])

if not os.path.exists(f"TextVectLayer_Vocab{VOCAB_SIZE}_maxlen{max_len}.h5"):
    hist = model.fit(train_dataset,epochs=EPOCHS,validation_data=test_dataset,
          callbacks=[Early])
    model.save(f"TextVectLayer_Vocab{VOCAB_SIZE}_maxlen{max_len}.h5")
    hist = hist.history
    with open(f"TextVectLayer_Vocab{VOCAB_SIZE}_maxlen{max_len}.pkl","wb") as f:
        pickle.dump(hist,f)
else:
    model = tf.keras.models.load_model(f"TextVectLayer_Vocab{VOCAB_SIZE}_maxlen{max_len}.h5")
    with open(f"TextVectLayer_Vocab{VOCAB_SIZE}_maxlen{max_len}.h5","rb") as f:
        hist = pickle.load(f)


x = np.arange(1,len(hist["loss"])+1)

plt.subplot(2,1,1)
plt.plot(x,hist["loss"],label="loss")
plt.plot(x,hist["val_loss"],label="val_loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot(x,hist["accuracy"],label="accuracy")
plt.plot(x,hist["val_accuracy"],label="val_accuracy")

plt.legend()
plt.show()