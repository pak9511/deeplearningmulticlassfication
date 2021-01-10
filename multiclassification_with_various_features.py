import numpy as np
from numpy import asarray
from numpy import zeros
import pandas as pd
import matplotlib.pyplot as plt
import re, nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Flatten, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, Conv2D, MaxPool2D,GRU
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from attention import BahdanauAttention
from tensorflow.keras.models import load_model

import pyTextMiner as ptm

def NLPpipeline(documents):

    result = pipeline.processCorpus(documents)

    text_data = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for _str in sent:
                if len(_str) > 0: new_doc.append(_str)
        text_data.append(new_doc)

    return text_data

def preprocess_text(sen):
  # Removing html tags
  sentence = re.sub(r'<[^>]+>', '', sen)
  # Remove punctuations and numbers
  sentence = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]+', ' ', sentence)
  # Single character removal
  sentence =  re.sub(r"\s+[ㄱ-ㅎㅏ-ㅣ가-힣]\s+", ' ', sentence)
  # Removing multiple spaces
  sentence = re.sub(r'\s+', ' ', sentence)

  return sentence

def pretrained_embedding_load(embedding_file):
  #embedding layer에 pre-trained glove model(100 hidden demension 사용) load
  embeddings_dictionary = dict()
  embedding = open(embedding_file, encoding='utf8')

  for line in embedding:
      records = line.split()
      word = records[0]
      try:
        vector_dimensions = asarray(records[1:], dtype='float32')
      except: continue
      embeddings_dictionary[word] = vector_dimensions

  embedding.close()

  num_demension = 300
  embedding_matrix = zeros((vocab_size, num_demension))
  for word, index in tokenizer.word_index.items():
      embedding_vector = embeddings_dictionary.get(word)
      if embedding_vector is not None:
          embedding_matrix[index] = embedding_vector

  return embedding_matrix

def pretrained_fasttest(model):
  # Getting the tokens
  from gensim.models.keyedvectors import KeyedVectors
  ko_model = KeyedVectors.load_word2vec_format(model, encoding='utf-8', unicode_errors='ignore')
  vector_dim = 300
  embeddings_dictionary = dict()
  # load fasttext vocab
  for word in ko_model.vocab:
    try:
      vector_dimension=asarray(ko_model[word], dtype='float32')
    except:continue
    embeddings_dictionary[word]=vector_dimension
  # mapping
  embedding_matrix = zeros((vocab_size, vector_dim))
  for word, index in tokenizer.word_index.items():
      embedding_vector = embeddings_dictionary.get(word)
      if embedding_vector is not None:
          embedding_matrix[index] = embedding_vector

  return embedding_matrix

def plot_graph(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_' + metric])

  if metric == 'acc':
    plt.title('model ' + 'accuracy')
  else:
    plt.title('model ' + metric)
  plt.ylabel(metric)
  plt.xlabel('epoch')
  plt.legend(['train','test'], loc='upper left')
  plt.show()


if __name__ == '__main__':

  mode = 'train'  
  selected_layer = 'cnn'  # bilstm, bilstmWithAttention, cnn 중 택1

  pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                          ptm.tokenizer.Komoran(),
                          # ptm.helper.POSFilter('NN*|JJ*'),
                          ptm.helper.SelectWordOnly(),
                          ptm.helper.StopwordFilter(file='stopwordsKor.txt')
                          )


  if mode == 'train':
    input_file = 'input.txt'
    topic_num=5
   
    df = pd.read_csv(input_file, encoding='utf-8', engine='python', header=0,index_col=0,sep='\t')

    df.dropna(how='any')
    print(df.dtypes)
    print(df['topic'].value_counts())
    print(df['sentiment'])
    print(df['text'])
    print(df['level'].value_counts())

    df['topic']=df['topic'].astype(int)
    y = df['level']

    # word로 표현된 label을 정수로 인코딩하는 작업(정수 0부터 시작하여, 라벨링 클래스 개수만큼 증가)
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y)

    # data set을 train과 text data set으로 분류
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # train과 test의 label을 one-hot encoding으로 전환
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X1_train = X_train["text"].tolist()   # X1_train["text"].tolist()) :: ['D1', 'D2', ... ]
    X1_test = X_test["text"].tolist()

    X1_train = NLPpipeline(X1_train)
    X1_test = NLPpipeline(X1_test)

    voca_size = 1000000  #the maximum number of words to keep, based on word frequency
    tokenizer = Tokenizer(num_words=voca_size)
    tokenizer.fit_on_texts(X1_train)

    X1_train = tokenizer.texts_to_sequences(X1_train)
    X1_test = tokenizer.texts_to_sequences(X1_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 256  #padding 작업을 위한 text의 maximum length 설정
    X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
    X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)

	#sentiment input
    X2_train = X_train['sentiment'].values 
    X2_test = X_test['sentiment'].values

	#topic input
    X3_train = X_train['topic'].values
    X3_test = X_test['topic'].values
    X3_train = to_categorical(X3_train)
    X3_test = to_categorical(X3_test)
    
    class_number = 3 #최종적으로 분류할 class 개수

    ## text data
    input_1 = Input(shape=(maxlen,)) # text input

    # fasttext
    embedding_matrix=pretrained_fasttest('./embeddings/cc.ko.300.vec')
    embedding_dim = 300
    embedding_layer = Embedding(output_dim=300, input_dim=vocab_size, weights=[embedding_matrix], input_length=maxlen,trainable=False)(input_1)

    # glove
    # embedding_matrix = pretrained_embedding_load('./embeddings/glove_korean_sns_300d.txt')
    # embedding_dim = 300
    # embedding_layer = Embedding(output_dim=300, input_dim=vocab_size, weights=[embedding_matrix], input_length=maxlen, trainable=False)(input_1)

    if selected_layer == 'bilstm':
      bilstm1 = Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3))(embedding_layer)
      layer = bilstm1

    elif selected_layer == 'bilstmWithAttention':
      lstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(
        LSTM(256, dropout=0.3, return_sequences=True, return_state=True,recurrent_activation='relu')
      )(embedding_layer)
      state_h = Concatenate()([forward_h, backward_h]) # 은닉 상태
      state_c = Concatenate()([forward_c, backward_c]) # 셀 상태
      attention = BahdanauAttention(128)# 가중치 크기 정의
      context_vector, attention_weights =  attention(lstm, state_h)
      layer = BatchNormalization()(context_vector)

    elif selected_layer == 'cnn':
      filter_sizes = 3  # convolutional filter 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) ex) "3" = trigram
      num_filters = 512  # filter의 수

      reshape = Reshape((maxlen, embedding_dim, 1))(embedding_layer)  # conv2d 함수가 요구하는 차원수로 만들어주기 위해 차원을 하나 추가(reshape)
      conv1 = Conv2D(num_filters, kernel_size=(filter_sizes, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
      maxpool1 = MaxPool2D(pool_size=(maxlen - filter_sizes + 1, 1), strides=(1, 1), padding='valid')(conv1)

      flatten = Flatten()(maxpool1)
      layer = flatten

    ## sentiment score data : 0 ~ 1 continuous numeric data
    input_2 = Input(shape=(1,))
    input_2_dense_layer_1 = Dense(10, activation='relu')(input_2)
    input_2_dense_layer_2 = Dense(10, activation='relu')(input_2_dense_layer_1)

    ## topics
    input_3= Input(shape=(topic_num,))
    input_3_dense_layer_1 = Dense(10, activation='relu')(input_3)
    input_3_dense_layer_2 = Dense(10, activation='relu')(input_3_dense_layer_1)

    ## concatenate three different layers
    concat_layer = Concatenate(axis=1)([layer, input_2_dense_layer_2,input_3_dense_layer_2])

    dense_layer = Dense(10, activation='relu')(concat_layer)
    output = Dense(class_number, activation='softmax')(dense_layer)
    model = Model(inputs=[input_1, input_2,input_3], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # 아키텍쳐 이미지 출력
    try:
      plot_model(model, to_file='./model_plot.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
      print(e)

    ## train model
    # validation_split=0.2을 주어서 훈련 데이터의 20%를 검증 데이터로 나누고, 검증 데이터를 보면서 훈련이 제대로 되고 있는지 확인. 검증 데이터는 기계가 훈련 데이터에 과적합 되고 있지는 않은지 확인하기 위한 용도로 사용됨.
    history = model.fit(x=[X1_train, X2_train,X3_train], y=y_train, batch_size=32, epochs=10, verbose=1, validation_split=0.2)

    ## save model
    model.save('./model.h5')

    ## evaluate model
    # 테스트 데이터에 대해 loss와 accuracy가 잘 나오는지 확인
    test_loss, test_acc = model.evaluate(x=[X1_test, X2_test,X3_test], y=y_test, verbose=1,batch_size=32)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)

    ## print the loss and accuracy for training and test sets
    plot_graph(history, 'acc')
    plot_graph(history, 'loss')