import preprocess as prep
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Flatten, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional, Conv2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.text import tokenizer_from_json
import tensorflow as tf
import numpy as np
import bert,os,json

def pretrained_embedding_load(embedding_file,vocab_size,num_demension,tokenizer):
    #embedding layer에 pre-trained glove model(100 hidden demension 사용) 로딩하는 함수
    embeddings_dictionary = dict()
    # pre-trained glove model의 단어와 임베딩 벡터값 읽어옴
    with open(embedding_file,encoding='utf-8') as embedding:
        for line in embedding:
          records = line.split()
          word = records[0]
          vector_dimensions = np.asarray(records[1:], dtype='float32')
          embeddings_dictionary[word] = vector_dimensions

    # 텍스트 데이터의 word_list와 로딩한 glove 모델의 word_list를 매핑
    embedding_matrix = np.zeros((vocab_size, num_demension))
    for word, index in tokenizer.word_index.items():
      embedding_vector = embeddings_dictionary.get(word)
      if embedding_vector is not None:
          embedding_matrix[index] = embedding_vector

    return embedding_matrix

def train(data,batch_size,epoch,maxlen,output_dir,selected_layer):

    # 한국어, 영어, 숫자만 텍스트에 남기고 형태소 분석 수행함
    cleaned = prep.preprocess(data['txt'].values)

    if selected_layer=='bert':
        # multi_cased 모델의 vocab 파일을 불러와 tokenizer 생성
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        tokenizer = FullTokenizer('./multi_cased_L-12_H-768_A-12/vocab.txt', do_lower_case=False)

        # 전처리된 텍스트를 BERT의 input 형태에 맞게 변환
        train_tokens = [["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in cleaned]
        train_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in train_tokens]
        train_data = pad_sequences(train_tokens_ids, maxlen=maxlen, dtype="long", truncating="post",
                                         padding="post")
        # bert_layer에 들어갈 input 형태 지정
        input_1 = Input(shape=(maxlen,), dtype=tf.int32, name="input_word_ids")

        # pre-trained 된 BERT 모델을 keras 레이어의 형태로 불러옴
        bert_params = bert.params_from_pretrained_ckpt('./multi_cased_L-12_H-768_A-12')
        bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
        # bert_layer에 input 레이어를 넣은 후, 신경망 레이어를 flatten하여 가중치를 보존하면서 2차원의 형태로 변환시킴
        bert_l = bert_layer(input_1)
        flatten = Flatten()(bert_l)
        layer=flatten

    else:
        # cnn, bilstm을 선택한 경우, tokenizer가 가질 최대 단어 개수 지정(voca_size) 후 tokenizer 생성
        voca_size = 1000000
        tokenizer = Tokenizer(num_words=voca_size)
        tokenizer.fit_on_texts(cleaned)
        # eval, predict 시 빠르게 불러오기 위해 tokenizer를 json 파일으로 저장
        tokenizer_json=tokenizer.to_json()
        with open('tokenizer.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        # 전처리된 텍스트를 정수로 변환한 후, 길이를 maxlen에 맞춤
        train_data = tokenizer.texts_to_sequences(cleaned)
        train_data = pad_sequences(train_data, padding='post', maxlen=maxlen)

        ## cnn, bilstm layer 에 들어갈 input 형태 지정
        input_1 = Input(shape=(maxlen,))

        # pre-trained glove 임베딩 모델 로딩
        vocab_size = len(tokenizer.word_index) + 1
        embedding_dim = 100
        embedding_matrix = pretrained_embedding_load(
            'glove.txt', vocab_size=vocab_size,
            num_demension=embedding_dim, tokenizer=tokenizer)
        embedding_layer = Embedding(output_dim=embedding_dim, input_dim=vocab_size, weights=[embedding_matrix],
                                    input_length=maxlen, trainable=False)(input_1)

        if selected_layer == 'bilstm':
            # 위에서 생성한 embedding_layer를 가져와서 bi-lstm 레이어에 넣음
            bilstm1 = Bidirectional(LSTM(256, dropout=0.3, recurrent_dropout=0.3))(embedding_layer)
            layer = bilstm1

        elif selected_layer == 'cnn':
            filter_sizes = 3  # convolutional filter 사이즈 지정, 3개의 단어를 보는것으로 지정함
            num_filters = 512  # filter의 수

            # conv2d 함수가 요구하는 차원수로 만들어주기 위해 차원을 하나 추가함(reshape)
            reshape = Reshape((maxlen, embedding_dim, 1))(embedding_layer)
            # 합성곱층과 풀링층을 거치면서 cnn레이어 구축
            conv1 = Conv2D(num_filters, kernel_size=(filter_sizes, embedding_dim), padding='valid',
                           kernel_initializer='normal',
                           activation='relu')(reshape)
            maxpool1 = MaxPool2D(pool_size=(maxlen - filter_sizes + 1, 1), strides=(1, 1), padding='valid')(conv1)
            # 이진 분류를 위해 2차원으로 레이어를 flatten함
            flatten = Flatten()(maxpool1)
            layer = flatten

    # 데이터의 라벨을 numpy 배열의 형태로 변환
    label=np.array(data['label'])

    # 은닉층의 출력 뉴런수를 줄이기 위해 relu 활성 함수 사용함
    dense_layer = Dense(16, activation='relu')(layer)
    # 과적합 방지를 위해 drop out 수행
    drop = Dropout(rate=0.1)(dense_layer)
    # 이진 분류이므로 출력 뉴런의 수를 1로 설정하고, sigmoid 활성 함수 사용함
    output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=input_1, outputs=output)
    #이진 분류이므로 loss function로는 binary_crossentropy, optimizer로는 adam 사용함
    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    print(model.summary())

    if selected_layer=='bert':
        # 모델 가중치 저장위해 callback 생성
        checkpointName = os.path.join(output_dir, "bert_model.ckpt")
        cp_callback = ModelCheckpoint(filepath=checkpointName, save_weights_only=True, verbose=1)
        model.fit(x=train_data, y=label, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2,callbacks=[cp_callback])
    else:
        model.fit(x=train_data, y=label, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.2)

    model.save(output_dir)

def eval(data,batch, maxlen, selected_layer,model_dir):

    cleaned_test = prep.preprocess(data['txt'].values)

    if selected_layer=='bert':
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        tokenizer = FullTokenizer('./multi_cased_L-12_H-768_A-12/vocab.txt', do_lower_case=False)

        eval_tokens = [["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in cleaned_test]
        eval_data = [tokenizer.convert_tokens_to_ids(token) for token in eval_tokens]
        eval_data = pad_sequences(eval_data, maxlen=maxlen, dtype="long", truncating="post", padding="post")

    else:
        with open('tokenizer.json') as f:
            json_data = json.load(f)
            tokenizer=tokenizer_from_json(json_data)
        eval_data = tokenizer.texts_to_sequences(cleaned_test)
        eval_data = pad_sequences(eval_data, padding='post', maxlen=maxlen)

    labels = np.array(data['label'])
    model= load_model(model_dir)

    test_loss, test_acc = model.evaluate(x=eval_data, y=labels, verbose=1, batch_size=batch)
    print("Test Loss: {}\nTest Accuracy:{}".format(test_loss,test_acc))

def predict(data, maxlen, model_dir,output_file_name):

    cleaned_predict= prep.preprocess(data['txt'].values)

    if selected_layer=='bert':
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        tokenizer = FullTokenizer('./multi_cased_L-12_H-768_A-12/vocab.txt', do_lower_case=False)

        predict_tokens = [["[CLS]"] + tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in cleaned_predict]                
        predict_data = [tokenizer.convert_tokens_to_ids(token) for token in predict_tokens]                                                                                                                                                   
        predict_data = pad_sequences(predict_data, maxlen=maxlen, dtype="long", truncating="post", padding="post")

    else:
        with open('tokenizer.json') as f:
            json_data = json.load(f)
            tokenizer=tokenizer_from_json(json_data)
            
        predict_data = tokenizer.texts_to_sequences(cleaned_predict)
        predict_data = pad_sequences(predict_data, padding='post', maxlen=maxlen)

    model = load_model(model_dir)
    # 각 데이터의 텍스트, 확률, 라벨을 예측하여 파일에 저장함
    result = model.predict(predict_data)
    label = np.around(model.predict(predict_data))
    with open(output_file_name, 'w', encoding='utf-8') as fw:
        for i in range(len(data)):
            fw.write('{}\t{}\t{}\n'.format(data['txt'].iloc[i], result[i],label[i]))

if __name__ == '__main__':
    # dataset.csv 파일의 단어 빈도수 상위 n개의 단어를 확인하기 위한 함수
    # prep.word_frequency('dataset.csv',100)

    # train, eval, predict에 사용할 데이터셋을 구성하는 함수. 랜덤샘플링을 통해 데이터를 분리함.
    train_data,eval_data,predict_data=prep.split_dataset('dataset.csv')

    # train, eval, predict 중 하나를 선택하여 mode로 지정함. mode를 분리하여 실험 파라미터 설정 및 수정을 용이하게하기 위함.
    mode='train'
    selected_layer = 'bert'  # cnn, bilstm, bert 중 택 1
    batch,epoch,maxlen=32,5,32
    model_dir = '.\\res\\' + selected_layer  ## cnn, bilstm의 경우 '.\\res\\'+selected_layer+'\\'+selected_layer+'_pos_model.h5'
    if mode=='train':
        train(data=train_data,batch_size=batch,epoch=epoch,maxlen=maxlen,
              output_dir=model_dir,selected_layer=selected_layer)

    elif mode=='eval':
        eval(data=eval_data,batch=batch,maxlen=maxlen,selected_layer=selected_layer,model_dir=model_dir)

    elif mode=='predict':
        output_file_name='result.txt' #predict 결과 파일
        predict(data=predict_data,maxlen= maxlen, model_dir=model_dir,output_file_name= output_file_name)