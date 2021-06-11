import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

dataset,info = tfds.load("fashion_mnist",as_supervised=True,with_info=True,split=["train","test"])

train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples

train_dataset = dataset[0]
test_dataset = dataset[1]

image, = train_dataset.take(1)

image_shape = image[0].numpy().shape
num_outputs = info.features["label"].num_classes
batch_size = 100
EPOCHS = 200

def img_norm(img,label):
    img = tf.cast(img,tf.float32)
    img = img/255
    return img,label

train_dataset = train_dataset.map(img_norm)
test_dataset = test_dataset.map(img_norm)

train_dataset = train_dataset.repeat().shuffle(train_size//4).batch(batch_size).prefetch(1)
test_dataset = test_dataset.batch(batch_size).prefetch(1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(image_shape[:-1])))
model.add(tf.keras.layers.Dense(units=512,activation="relu"))
model.add(tf.keras.layers.Dense(units=256,activation="relu"))
model.add(tf.keras.layers.Dense(units=128,activation="relu"))
model.add(tf.keras.layers.Dense(units=64,activation="relu"))
model.add(tf.keras.layers.Dense(units=num_outputs,activation="softmax"))

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"]>0.99:
            self.model.stop_training = True

Early = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
CustomCallback = MyCallback()
calls = [Early,CustomCallback]

model.fit(train_dataset,
          epochs=EPOCHS,
          steps_per_epoch=train_size//batch_size,
          validation_data=test_dataset,
         validation_steps=test_size//batch_size,
          callbacks=calls)

model.save("FASHION_MNIST/MNIST_DNN.h5")
