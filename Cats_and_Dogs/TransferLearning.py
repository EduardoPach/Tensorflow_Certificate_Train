import tensorflow_hub as hub
import tensorflow as tf
import os
import pickle

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

URL_mobilenet = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
URL_inception = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
print(hub.__version__)

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
print("-"*50)
print("="*50)

IMG_SIZE = 224
BATCH_SIZE = 35
EPOCHS = 5
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,
                                                                  horizontal_flip=True,
                                                                  vertical_flip=True)

validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

train_data = train_generator.flow_from_directory(directory=train_dir,
                                                 target_size=(IMG_SIZE,IMG_SIZE),
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=True,
                                                 class_mode="binary")

valid_data = validation_generator.flow_from_directory(directory=validation_dir,
                                                      target_size=(IMG_SIZE,IMG_SIZE),
                                                      batch_size=BATCH_SIZE,
                                                      class_mode="binary")

feature_extractor = hub.KerasLayer(URL_mobilenet,input_shape=(IMG_SIZE,IMG_SIZE,3))
feature_extractor.trainable=False

model = tf.keras.Sequential()
model.add(feature_extractor)
model.add(tf.keras.layers.Dense(num_outputs,activation="softmax"))

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

if  not os.path.exists("Transferlearning.h5"):
    hist = model.fit(train_data,
                        epochs=EPOCHS,
                        steps_per_epoch=(num_train_dogs+num_train_cats)//BATCH_SIZE,
                        validation_data=valid_data,
                        validation_steps=(num_validation_cats+num_validation_dogs)//BATCH_SIZE)
    tf.saved_model.save(model, "SavedTransferLearning.h5")
    model.save("Transferlearning.h5")
    with open("TransferLearning.pkl", "wb") as f:
        pickle.dump(hist.history, f)

else:
    model = tf.keras.models.load_model("Transferlearning.h5",
                                       custom_objects={"KerasLayer":hub.KerasLayer})

    model.summary()