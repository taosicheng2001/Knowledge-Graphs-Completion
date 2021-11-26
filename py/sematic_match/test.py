import numpy as np
import math

def test_score():

    entity_vector_Dict = np.load('../dataset/entity_vector_Dict.npy',allow_pickle='TRUE').item()
    relation_vector_Dict = np.load('../dataset/relation_vector_Dict.npy',allow_pickle='TRUE').item()
    cnt = -1
    with open('../dataset/train.txt','r') as f:
        tuples = f.readlines()
        for tuple in tuples:
            cnt += 1
            print("Testing tuple: " + str(cnt))
            tuple = tuple.split('\t')
            tuple[-1] = tuple[-1].rstrip('\n')
            if( (tuple[0] in entity_vector_Dict.keys()) and (tuple[2] in entity_vector_Dict.keys())):
                vector_head = entity_vector_Dict[tuple[0]]
                vector_tail = entity_vector_Dict[tuple[2]]
                if(tuple[1] in relation_vector_Dict.keys()):
                    score_sqare = 0
                    vector_relation = relation_vector_Dict[tuple[1]]
                    for i in range(0,256):
                        vector_tail[i] -= (vector_head[i] + vector_relation[i])
                        score_sqare += vector_tail[i]*vector_tail[i]
                    score = math.sqrt(score_sqare)
                    #if(score < 10000):
                    #    print(cnt)
                    print(score)    
                else:
                    print("No Relation")
                    exit()
            else:
                print("No Entity")

if __name__ == '__main__':
    test_score()