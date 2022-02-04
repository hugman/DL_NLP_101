"""
    Attention : Number dataset 


    Author : Sangkeun Jung (2021, hugmanskj@gmail.com)
    
    Copyright(c) 2021 All rights reserved.
"""

#----------------------------------
# Number Dataset 설명
#----------------------------------
# 입력 데이터 
#    : Xs => 최소 5자리 최대 20자리 미만의 숫자들의 나열
#    : q  => Xs와 비교할 수 있는 query 숫자 하나

# 출력 데이터
#    : y  => Xs 의 숫자들 중 q보다 큰 최초의 수 혹은 그 사이의 숫자
#    why 사이의 숫자? --> to verify the blendding effect not a simple pointing effect. 


# 예 1)
#    Xs : 3, 5, 6, 2, 5, 7, 2
#    q  : 4
#    Y  : 5   (Xs 에서, 4보다 큰 최초의 숫자는 5가 됨)

# 예 2)
#    Xs : 1, 4, 7, 9, 8
#    q  : 4
#    Y  : 6   
#
# 위 문제를 풀 수 있으려면, neural network 이 query 의 값보다 큰 최초의 숫자에 대해 attention 할 수 있어야 한다. 

import os 
import numpy as np
np.random.seed(42)
import random 
random.seed(42)

import os, sys
os.chdir( os.path.dirname( os.path.abspath(__file__ ) ) )

def generate_data(num_examples):

    min_digit = 5
    max_digit = 10

    min_number = 0
    max_number = 10

    data = []
    for i in range(num_examples):
        while True:
            # sequence generation (Xs)
            seq = np.random.randint(min_number, high=max_number, size=10)
            n_digit = np.random.randint(min_digit, high=max_digit, size=1)[0]
            seq = seq[:n_digit]
        
            # query generation (q)
            min_number_in_seq = np.min(seq)
            max_number_in_seq = np.max(seq)

            if min_number_in_seq == max_number_in_seq: continue 
            query = np.random.randint(min_number_in_seq, high=max_number_in_seq, size=1)[0]
            break


        # output generation (y)
        candidates = [ (pos, num) for pos, num in enumerate(seq) if num > query] 
        candidates = sorted(candidates, key=lambda x: x[1])
        y_tuple = candidates[0]  
        pos_y, num_y = y_tuple

        y_s = list( range(query+1, num_y))
        if len(y_s) == 0: # case --> y = num_y
            y = num_y
        else:
            y = y_s[-1]

        data.append( (list(seq), query, y) )
        #print(seq)
        #print(query)
        #print(num_y)
        #print(y_s)
        #print(y)
        #print("---")
    return data 


def dump_data(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for seq, query, y in data:
            seq_str = ",".join( [str(s) for s in seq] )
            print("{}\t{}\t{}".format(seq_str, query, y), file=f)

        print("# of examples : ", len(data))
        print("Data is dumped at ", fn)

def dump_data_sequence_form(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        for seq, query, y in data:
            seq_str = ",".join( [str(s) for s in seq] )
            print(f"{query}|{seq_str}\t{y}", file=f)

        print("# of examples : ", len(data))
        print("Data is dumped at ", fn)


if __name__ == '__main__':
    data_root = './data/numbers'
    os.makedirs(data_root, exist_ok=True)

    fns = {
                'train' : os.path.join(data_root, 'train.txt'),
                'test'  : os.path.join(data_root, 'test.txt'),

                'train_seq' : os.path.join(data_root, 'train.seq.txt'),
                'test_seq'  : os.path.join(data_root, 'test.seq.txt'),

          }

    if not os.path.exists(data_root): os.makedirs(data_root)


    all_data = generate_data(num_examples=50000)

    train_data = all_data[:45000]
    test_data = all_data[45000:]

    dump_data(train_data, fns['train'])
    dump_data(test_data, fns['test'])

    dump_data_sequence_form(train_data, fns['train_seq'])
    dump_data_sequence_form(test_data, fns['test_seq'])





