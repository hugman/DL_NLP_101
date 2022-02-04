"""
    Attention : Number dataset 

    for sorting

    Author : Sangkeun Jung (2021, hugmanskj@gmail.com)
    
    Copyright(c) 2021 All rights reserved.
"""

#----------------------------------
# Number Dataset 설명
#----------------------------------
# 입력 데이터 : random number
# 출력 데이터 : sorted random number (removing duplicated numbers)


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
            sorted_seq = sorted( list(set(seq)) ) 
            seq_str = ",".join( [str(s) for s in seq] )
            sorted_seq_str = ",".join( [str(s) for s in sorted_seq] )
            print("{}\t{}".format(seq_str, sorted_seq_str), file=f)

        print("# of examples : ", len(data))
        print("Data is dumped at ", fn)
if __name__ == '__main__':
    data_root = './data/sorted_numbers'
    os.makedirs(data_root, exist_ok=True)

    fns = {
                'train' : os.path.join(data_root, 'train.txt'),
                'test'  : os.path.join(data_root, 'test.txt'),
          }

    os.makedirs(data_root, exist_ok=True)


    all_data = generate_data(num_examples=50000)

    train_data = all_data[:45000]
    test_data = all_data[45000:]

    dump_data(train_data, fns['train'])
    dump_data(test_data, fns['test'])




