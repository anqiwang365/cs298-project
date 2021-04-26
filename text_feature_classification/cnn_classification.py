
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Model
from sklearn.metrics import classification_report

MAX_SEQUENCE_LENGTH = 267
EMBEDDING_DIM = 300

########## Data Pre-processing ############
# read data 0:negative 1:positive
def loaddata(filename):
    data = pd.read_csv(filename)
    return data

# clean data, remove punctuation
def remove_punct(text):
    text_nopunct = re.sub('[' + string.punctuation + ']', '', text)
    return text_nopunct

# clean data
def preprocess_data(data):
    tokens = []
    # 1. remove url
    data = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', data)
    # 2. remove punctuation
    data = data.translate(str.maketrans('', '', string.punctuation))
    # 3. remove emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)
    # 4. remove '#mkr
    data = data.replace('mkr', '')
    # 5. replace multiple space with single space
    data = re.sub(r'\s+', ' ', data)
    # 6. convert to lower case
    data = data.lower()
    return data


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=False)

    sequence_input = layers.Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    convs = []
    filter_sizes = [2, 3, 4, 5, 6]
    for filter_size in filter_sizes:
        l_conv = layers.Conv1D(filters=200,
                               kernel_size=filter_size,
                               activation='relu')(embedded_sequences)
        l_pool = layers.GlobalMaxPooling1D()(l_conv)
        convs.append(l_pool)
    l_merge = layers.concatenate(convs, axis=1)
    x = layers.Dropout(0.1)(l_merge)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    preds = layers.Dense(labels_index, activation='sigmoid')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    return model

def main():
    data = loaddata('collectdata.csv')
    # data = loaddata('tweet_dataset_edited.csv')
    data['Text_Clean'] = data['tweet_content'].apply(lambda x: preprocess_data(x))
    tokens = [word_tokenize(sen) for sen in data.Text_Clean]


    # remove stopwords
    stoplist = stopwords.words('english')
    filtered_words = [[word for word in sen if word not in stoplist] for sen in tokens]
    data['Text_Final'] = [' '.join(sen) for sen in filtered_words]
    data['tokens'] = filtered_words

    # Split data into neg and pos
    pos = []
    neg = []
    for l in data.category:
        if l == 0: #neg
            pos.append(0)
            neg.append(1)
        elif l == 1: #pos
            pos.append(1)
            neg.append(0)
    data['Normal'] = pos
    data['Bullying'] = neg

    data = data[['Text_Final', 'tokens', 'category', 'Normal', 'Bullying']]

    # 80% training and 20% testing dataset
    data_train, data_test = train_test_split(data, test_size=0.20, random_state=42)

    # build training vocabulary and get maximum training sentence length
    # and total number of words training data.
    all_training_words = [word for tokens in data_train["tokens"] for word in tokens]
    training_sentence_lengths = [len(tokens) for tokens in data_train["tokens"]]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))

    # build testing vocabulary and get maximum testing sentence length and total number of words in testing data
    all_test_words = [word for tokens in data_test['tokens'] for word in tokens]
    test_sentence_lengths = [len(tokens) for tokens in data_test['tokens']]
    TEST_VOCAB = sorted(list(set(all_test_words)))

    # Loading Google News Word2Vec model
    word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
    word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    # Tokenize and Pad sequences
    tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
    tokenizer.fit_on_texts(data_train['Text_Final'].tolist())
    training_sequences = tokenizer.texts_to_sequences(data_train['Text_Final'].tolist())
    train_word_index = tokenizer.word_index
    train_cnn_data = pad_sequences(training_sequences,
                                   maxlen=MAX_SEQUENCE_LENGTH)

    # get embeddings from Google News Word2Vec model
    train_embedding_weights = np.zeros((len(train_word_index) + 1, 300))
    for word, index in train_word_index.items():
        train_embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(300)


    test_sequences = tokenizer.texts_to_sequences(data_test["Text_Final"].tolist())
    test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    label_names = ['Normal', 'Bullying']
    y_train = data_train[label_names].values
    x_train = train_cnn_data
    y_tr = y_train
    model = ConvNet(train_embedding_weights,
                    MAX_SEQUENCE_LENGTH,
                    len(train_word_index) + 1,
                    EMBEDDING_DIM,
                    len(list(label_names)))
    #training
    num_epochs = 3
    batch_size = 32
    hist = model.fit(x_train,
                     y_tr,
                     epochs=num_epochs,
                     validation_split=0.1,
                     shuffle=True,
                     batch_size=batch_size)

    # testing
    predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
    labels = [1, 0]
    prediction_labels = []
    for p in predictions:
        prediction_labels.append(labels[np.argmax(p)])
    accuracy = sum(data_test.category == prediction_labels) / len(prediction_labels)

    print('Accuracy:', accuracy)

    y_pred_bool = np.argmax(predictions, axis=1)
    print(classification_report(data_test.category, prediction_labels))

if __name__ == "__main__":
    main()