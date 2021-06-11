import tensorflow as tf
import os
import pandas as pd
import numpy as np
import string

n_songs = 50
df = pd.read_csv("C:\\Users\\acer\\.keras\\datasets\\songdata.csv",dtype=str)[:n_songs]

def tokenize_corpus(corpus, num_words=-1):
  # Fit a Tokenizer on the corpus
  if num_words > -1:
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
  else:
    tokenizer = tf.keras.preprocessing.text.Tokenizer()

  tokenizer.fit_on_texts(corpus)
  return tokenizer

def create_lyrics_corpus(dataset, field):
  # Remove all other punctuation
  dataset[field] = dataset[field].str.replace('[{}]'.format(string.punctuation), '')
  # Make it lowercase
  dataset[field] = dataset[field].str.lower()
  # Make it one long string to split by line
  lyrics = dataset[field].str.cat()
  corpus = lyrics.split('\n')
  # Remove any trailing whitespace
  for l in range(len(corpus)):
    corpus[l] = corpus[l].rstrip()
  # Remove any empty lines
  corpus = [l for l in corpus if l != '']

  return corpus

corpus = create_lyrics_corpus(df,'text')
tokenizer = tokenize_corpus(corpus)


total_words = len(tokenizer.word_index)+1

seq = list()

for line in corpus:
    tok_seq = tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(tok_seq)):
        n_gram = tok_seq[:i+1]
        seq.append(n_gram)

max_len = max([len(i) for i in seq])

seq = tf.keras.preprocessing.sequence.pad_sequences(sequences=seq,
                                                    maxlen=max_len,
                                                    padding="pre")

seq = np.array(seq)
input_seq,labels = seq[:,:-1],seq[:,-1]
labels = tf.keras.utils.to_categorical(labels)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=total_words,output_dim=64,input_length=max_len-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)))
model.add(tf.keras.layers.Dense(total_words,activation="softmax"))

model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])

if not os.path.exists(f"LSTM_songs{n_songs}.h5"):
    hist = model.fit(input_seq,labels,epochs=200)
    model.save(f"LSTM_songs{n_songs}.h5")
else:
    model = tf.keras.models.load_model(f"LSTM_songs{n_songs}.h5")

seed_text = "welcome to the jungle"
next_words = 20

for i in range(200):
    tok_list = tokenizer.texts_to_sequences([seed_text])[0]
    pad_list = tf.keras.preprocessing.sequence.pad_sequences(sequences=[tok_list],
                                                             maxlen=max_len-1,
                                                             padding="pre")
    pred_prob = model.predict(pad_list)[0]
    pred = np.random.choice([x for x in range(len(pred_prob))],
                            p=pred_prob)
    output=""
    for word,index in tokenizer.word_index.items():
        if index == pred:
            output=word
            break
    seed_text += " "+output

print(seed_text)