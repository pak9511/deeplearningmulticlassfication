import pandas as pd
from sklearn.model_selection import train_test_split
from konlpy.tag import Komoran
komoran = Komoran()

def split_dataset(file_path):
    # Train,Eval 데이터로 쓸 추가 파일 불러옴
    df_add = pd.read_csv('corpus.csv', encoding='utf-8', header=0)

    # dataset.csv 파일을 불러오고 데이터 중복제거 실시
    df = pd.read_csv(file_path, sep=',', encoding='utf-8', header=0, index_col=0)
    df = df.drop_duplicates()

    # 빈도수가 높은 단어 중 영화 관련 단어와 게임 관련 단어를 선정함
    word_list_movie = ['영화', '연기', '드라마', '배우', '감독', '감동', '작품', '액션', '명작']
    word_list_game = ['게임', '업데이트', '메이플', '계정', '연동', '아이폰', '접속', '메이플스토리', '레벨', '퀘스트']

    # dataset.csv 파일에서 영화 관련 단어가 등장하는 텍스트와 게임 관련 단어가 등장하는 텍스트를 분리하고 각각 1, 0의 라벨을 부여함
    extract_movie, extract_game,predict = [], [],[]
    for e, txt in enumerate(df['txt'].values):
        for noun in komoran.nouns(txt):
            if noun in word_list_movie:
                extract_movie.append(txt)
                break
        for noun in komoran.nouns(txt):
            if noun in word_list_game:
                extract_game.append(txt)
                break
    df_movie = pd.DataFrame(extract_movie,columns=['txt'])
    df_movie['label']=1
    df_game = pd.DataFrame(extract_game,columns=['txt'])
    df_game['label']=0

    df_total=pd.concat([df_add,df_movie,df_game])

    # dataset.csv에서 영화, 게임 관련 단어가 포함되지 않은 텍스트를 분리하여 predict 시에 사용
    for txt in df['txt'].values:
        if txt not in extract_movie and txt not in extract_game:
            predict.append(txt)
    df_predict = pd.DataFrame(predict, columns=['txt'])

    train, eval = train_test_split(df_total, test_size=0.20, random_state=42)
    print('The number of train set: {} \nThe number of eval set: {} \nThe number of predict set: {}'.format(train.shape[0],eval.shape[0],len(predict)))

    return train,eval,df_predict

def preprocess(text):
    import re
    sentence=[]
    for sen in text:
        # 숫자, 영어, 한글 외의 특수 문자 제거
        sen=re.sub('[^0-9a-zA-zㄱ-ㅎㅏ-ㅣ가-힣]+', ' ', str(sen))
        # 특수 문자 제거 후 비어있지 않은 데이터들만 형태소 분석 수행
        if re.fullmatch(r'[\s]+', sen)==None:
            sen_morph=' '.join(komoran.morphs(sen))
        else:
            sen_morph="0"
        sentence.append(sen_morph)
    return sentence

def word_frequency(file_path,n):
    from collections import Counter

    df = pd.read_csv(file_path, sep=',', encoding='utf-8', header=0, index_col=0)

    with open('한국어불용어100.txt', encoding='utf-8') as fr:
        stop = [i.split('\t')[0] for i in fr]

    extract, freq = [], []
    for e, txt in enumerate(df['txt'].values):
        for noun in komoran.nouns(txt):
            if noun not in stop:
                extract.append(noun)
        counter = Counter(extract)

    for word, count in counter.most_common(n):
        freq.append((word, count))

    df2 = pd.DataFrame(freq)
    df2.to_csv('word_freq.txt', encoding='utf-8', header=None, index=False)