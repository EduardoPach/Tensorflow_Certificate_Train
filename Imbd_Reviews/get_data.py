import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def get_imbd_reviews(vocab_size=2000,seq_size=50,pad='post',trunc='post',size_train=10000,size_test=5000):
    dataset,info = tfds.load("imdb_reviews",
                            as_supervised=True,
                            with_info=True,
                            shuffle_files=True)


    train_dataset = dataset['train']
    test_dataset = dataset['test']
    unsupervised_dataset = dataset['unsupervised']

    num_train = info.splits["train"].num_examples
    num_test = info.splits["test"].num_examples
    num_unsupervised = info.splits["unsupervised"].num_examples

    train_examples = []
    train_labels = list()
    test_examples = []
    test_labels = list()
    unsupervised_examples = list()

    for text,label in train_dataset.take(size_train):
        train_labels.append(label.numpy())
        train_examples.append(str(text.numpy()))

    for text,label in test_dataset.take(size_test):
        test_labels.append(label.numpy())
        test_examples.append(str(text.numpy()))

    for text,_ in unsupervised_dataset.take(1000):
        unsupervised_examples.append(str(text.numpy()))


    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>",num_words=vocab_size)
    tokenizer.fit_on_texts(train_examples)

    train_seq = tokenizer.texts_to_sequences(train_examples)
    test_seq = tokenizer.texts_to_sequences(test_examples)
    unsupervised_seq = tokenizer.texts_to_sequences(unsupervised_examples)

    train_final = tf.keras.preprocessing.sequence.pad_sequences(sequences=train_seq,
                                                                maxlen=seq_size,
                                                                padding=pad,
                                                                truncating=trunc)

    test_final = tf.keras.preprocessing.sequence.pad_sequences(sequences=test_seq,
                                                                maxlen=seq_size,
                                                                padding=pad,
                                                                truncating=trunc)

    unsupervised_final = tf.keras.preprocessing.sequence.pad_sequences(sequences=unsupervised_seq,
                                                                    maxlen=seq_size,
                                                                    padding=pad,
                                                                    truncating=trunc)

    return (train_final,np.array(train_labels)),(test_final,np.array(test_labels)),unsupervised_final