import tensorflow as tf
import tensorflow_datasets as tfds
import os
import pickle

dataset,info = tfds.load("fashion_mnist",
                                              as_supervised=True,
                                              with_info=True,
                                              split=["train","test"])

train_dataset, test_dataset = dataset[0],dataset[1]

IMG_SIZE = 28
BATCH_SIZE = 100
EPOCHS = 100
train_size = info.splits["train"].num_examples
test_size = info.splits["test"].num_examples
num_output = info.features["label"].num_classes

print(train_dataset)

def image_resize_norm(images,labels):
    images = tf.image.resize(images,(IMG_SIZE,IMG_SIZE))/255.0
    return images,labels

train_dataset = train_dataset.map(image_resize_norm).shuffle(train_size//4).batch(BATCH_SIZE).prefetch(1)
test_dataset = test_dataset.map(image_resize_norm).batch(BATCH_SIZE).prefetch(1)

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] >= 0.95:
            self.model.stop_training = True


Early = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
Custom = MyCallback()
calls = [Early,Custom]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=3,
                                 activation="relu",
                                 input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=2,activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64,activation="relu"))
model.add(tf.keras.layers.Dense(num_output,activation="softmax"))

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

if not os.path.exists("Fashion_MNIST_CNN.h5") or not os.path.exists(("Fashion_MNIST_CNN.pkl")):
    hist = model.fit(train_dataset,
                     epochs=EPOCHS,
                     validation_data = test_dataset,
                     callbacks=calls)

    model.save("Fashion_MNIST_CNN.h5")
    hist = hist.history
    with open("Fashion_MNIST_CNN.pkl","wb") as f:
        pickle.dumps(hist,f)
else:
    model = tf.keras.models.load_model("Fashion_MNIST_CNN.h5")
    with open("Fashion_MNIST_CNN.pkl",'rb') as f:
        hist = pickle.load(f)

loss = hist["loss"]
val_loss = hist["val_loss"]
acc = hist["accuracy"]
val_acc = hist["val_accuracy"]
x = np.arange(1,EPOCHS+2)

plt.subplot(2,1,1)
plt.plot(x,loss,label="loss")
plt.plot(x,val_loss,label="val_loss")
plt.ylabel("Loss")

plt.subplot(2,1,2)
plt.plot(x,acc,label="accuracy")
plt.plot(x,val_acc,label="val_accuracy")
plt.xlabel("EPOCH")
plt.ylabel("Accuracy")

plt.show()