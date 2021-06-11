import tensorflow as tf
import os
import pickle
keras = tf.keras

base_dir = "C:/Users/acer/.keras/datasets/cats_and_dogs_filtered"

train_dir = os.path.join(base_dir,"train")
validation_dir = os.path.join(base_dir,"validation")

train_cats_dir = os.path.join(train_dir,"cats")
train_dogs_dir = os.path.join(train_dir,"dogs")

validation_cats_dir = os.path.join(validation_dir,"cats")
validation_dogs_dir = os.path.join(validation_dir,"dogs")

num_train_cats = len(os.listdir(train_cats_dir))
num_train_dogs = len(os.listdir(train_dogs_dir))

num_validation_cats = len(os.listdir(validation_cats_dir))
num_validation_dogs = len(os.listdir(validation_dogs_dir))

num_outputs = len(os.listdir(base_dir))

print("="*50)
print(f'Number of cats in train: {num_train_cats}')
print(f'Number of cats in validation: {num_validation_cats}')
print(f'Number of dogs in train: {num_train_dogs}')
print(f'Number of dogs in validation: {num_validation_dogs}')
print("-"*50)
print(f'Total cats: {num_train_cats+num_validation_cats}')
print(f'Total dogs: {num_validation_dogs+num_train_dogs}')
print(f"Total pics: {num_train_dogs+num_validation_cats+num_validation_dogs+num_train_cats}")
print("="*50)

IMG_SIZE = 224
BATCH_SIZE = 35
EPOCHS = 200
generator = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_data = generator.flow_from_directory(directory=train_dir,
                                           target_size=(IMG_SIZE,IMG_SIZE),
                                           shuffle=True,
                                           batch_size=BATCH_SIZE,
                                           class_mode="binary")

valid_data = generator.flow_from_directory(directory=validation_dir,
                                                target_size=(IMG_SIZE,IMG_SIZE),
                                                batch_size=BATCH_SIZE,
                                                class_mode="binary")

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=64,
                              kernel_size=3,
                              activation="relu",
                              input_shape=(IMG_SIZE,IMG_SIZE,3)))

model.add(keras.layers.MaxPool2D(3))
model.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))
model.add(keras.layers.MaxPool2D(3))
model.add(keras.layers.Conv2D(filters=16,kernel_size=3,activation="relu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(512,activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(128,activation="relu"))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(64,activation="relu"))
model.add(keras.layers.Dense(num_outputs,activation="softmax"))

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

Early = keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

hist = model.fit_generator(train_data,
          epochs=EPOCHS,
          steps_per_epoch=(num_train_cats+num_train_dogs)//BATCH_SIZE,
          validation_data=valid_data,
          validation_steps=(num_validation_dogs+num_validation_cats)//BATCH_SIZE,
          callbacks=[Early])

model.save("CatsDogs_CNN.h5")
with open("CatsDogs_CNN.pkl",'wb') as f:
    pickle.dump(hist.history,f)


