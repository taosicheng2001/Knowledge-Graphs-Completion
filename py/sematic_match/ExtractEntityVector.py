from typing import OrderedDict
from gensim.models import KeyedVectors, word2vec , Word2Vec
import multiprocessing
import numpy as np
import math

def struct_Description_Dict():
    entity_word_Dict = {}
    sentences_list = []

    # read entity_data
    with open('../dataset/entity_with_text.txt','r') as f:
        entity_word_lists = f.readlines()
        for entity_word_list in entity_word_lists:
            # temp[0] = id
            # temp[1] = description
            
            temp = entity_word_list.split('\t')
            temp[1] = temp[1].rstrip('\n')

            print('Reading ID: ' + temp[0])

            # Add [Id, decription] to Dict
            entity_word_Dict[temp[0]] = temp[1]

            # Add Decription to Sentence_list
            sentences_list.append(temp[1])
    
    # save description
    with open('../dataset/entity_word_description.txt','w') as f2:
        for description_word in sentences_list:
            description_word = description_word + '\n'
            f2.write(description_word)
    # save entity_word_Dict
    np.save('../dataset/entity_word_description',entity_word_Dict)


def train_model():

    sentences = list(word2vec.LineSentence('../dataset/entity_word_description.txt'))
    model = Word2Vec(sentences,vector_size=256,min_count=1,window=5,sg=0,workers=multiprocessing.cpu_count(),max_vocab_size = None,sorted_vocab = 1)
    model.wv.save_word2vec_format('../dataset/word2vec.vector')


def read_model():

    entity_word_Dict = np.load('../dataset/entity_word_description.npy',allow_pickle='TRUE').item()
    entity_vector_Dict = {}
    
    # 初始化 entity_vecotr_Dict
    for id in entity_word_Dict.keys():
        zero_vector = []
        for i in range(0,256):
            zero_vector.append(1)
        entity_vector_Dict[id] = zero_vector
        print('create zero vector for: '+ id)
    
    print_vector(entity_vector_Dict)

    with open('../dataset/word2vec.vector','r') as f:
        vector_word_lists = f.readlines()
        for vector_word_list in vector_word_lists:
            temp = vector_word_list.split()
            print('Dealing with word: '+ temp[0])
            print(len(temp))
            size = len(temp)
            print(temp[1])
            for id in entity_vector_Dict.keys():
                # temp[0] in entity_vector description
                if temp[0] in entity_word_Dict[id]:
                    # 把temp[1-256]叠加到vector
                    for i in range(1,size):
                        (entity_vector_Dict[id])[i-1] *= float(temp[i])
                    print("ADD "+ temp[0] + " to entity " + id)

    #归一化
    for id in entity_vector_Dict.keys():
        vector = entity_vector_Dict[id]
        moor_sqare = 0
        for i in range(0,256):
            moor_sqare += vector[i]*vector[i]
        print(moor_sqare)
        if(moor_sqare == 0):
            continue
        moor = math.sqrt(moor_sqare)
        for i in range(0,256):
            entity_vector_Dict[id][i] = vector[i]/moor

    print_vector(entity_vector_Dict)
    np.save('../dataset/entity_vector_Dict',entity_vector_Dict)


def print_vector(Dict: dict):
    for key,value in Dict.items():
        print(key)
        print(value)    


if __name__ == '__main__':
    #struct_Description_Dict()
    #train_model()
    read_model()

