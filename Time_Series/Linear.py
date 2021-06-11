import tensorflow as tf
import os
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

keras = tf.keras

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)

csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
# slice [start:stop:step], starting from index 5 take every 6th record.
df = df[5::6]

cols = ['p (mbar)', 'T (degC)']

T = df[cols[1]].values

@tf.autograph.experimental.do_not_convert
def window_dataset(series, window_size, batch_size=32,
                   shuffle_buffer=10000):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


n = len(T)
split = int(n*0.8)

train_T = T[:split]
test_T = T[split:]


window_size = 24

train_dataset = window_dataset(train_T,window_size)
test_dataset = window_dataset(test_T,window_size)

model = keras.models.Sequential([
  keras.layers.Dense(1, input_shape=[window_size])
])

opt = keras.optimizers.Adam(lr=1e-4)

model.compile(optimizer=opt,loss="mse",metrics=["mae"])
EPOCHS = 200
Early = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
hist = model.fit(train_dataset,epochs=EPOCHS,validation_data=test_dataset,callbacks=[Early])




