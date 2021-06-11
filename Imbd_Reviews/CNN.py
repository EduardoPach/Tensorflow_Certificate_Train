import tensorflow as tf
import os
from get_data import get_imbd_reviews
import matplotlib.pyplot as plt
import pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

vocab_size=2000
seq_size=50
embed_size = 15

train_data,test_data,new_data = get_imbd_reviews()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embed_size,input_length=seq_size))
model.add(tf.keras.layers.Conv1D(16,5,activation="relu"))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


opt = tf.keras.optimizers.Adam(lr=0.01)
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])

hist = model.fit(train_data[0],train_data[1],epochs=100,validation_data=test_data)

model.save("CNN_imdb_reviews.h5")
with open("CNN_imdb_reviews.pkl","wb") as f:
    pickle.dump(hist.history, f)

x = np.arange(1,101)

plt.subplot(2,1,1)
plt.plot(x,hist.history["loss"],label="loss")
plt.plot(x,hist.history["val_loss"],label="val_loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot(x,hist.history["accuracy"],label="accuracy")
plt.plot(x,hist.history["val_accuracy"],label="val_accuracy")

plt.legend()
plt.show()