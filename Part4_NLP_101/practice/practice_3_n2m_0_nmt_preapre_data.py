"""
    NLP 101 > N2M Task (Neural Machine Translation) - Preprocessor

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""

## All data is from https://aihub.or.kr/aidata/87
## We only use sample data
##  - all *.xls files are integrated


## in this code, we will
##      - preapre two types of tokeziner for encoder(source language) and decoder(target language).
##      - prepare vocabularies and tokneized data
##      - the file is '*.xlsx'

in_fns = {
                'train': './data/translation/train.xlsx',
                'valid': './data/translation/valid.xlsx',
                'test':  './data/translation/test.xlsx',
        }
to_fns = {
            'source_vocab': './data/translation/source.vocab',
            'target_vocab': './data/translation/target.vocab',
            'vocab_info':   './data/translation/vocab.info.json',
         }

def dump_pkl(data, to_fn):
    import pickle as pkl 
    with open(to_fn, 'wb') as f:
        pkl.dump(data, f)
        print("[DUMP] data file is dumped at ", to_fn)


def prepare_data(in_fns, src_tokenizer, tar_tokenizer):
    import pandas as pd

    data_tags = ['train', 'test', 'valid']
    for tag in data_tags:
        print(f"Working on {tag} ------------------")
        df = pd.read_excel(in_fns[tag])[ ['source', 'target' ] ]
        src_data = src_tokenizer( df.source.values.tolist() )
        tar_data = tar_tokenizer( df.target.values.tolist() )

        ## merge data 
        keys = ['input_ids', 'token_type_ids', 'attention_mask']
        for key in keys:
            if key in src_data : df[f'source_{key}'] = src_data[key]
            if key in tar_data : df[f'target_{key}'] = tar_data[key]

        ## dump
        to_fn = f'./data/translation/{tag}.processed.pkl'
        df.to_pickle(to_fn)
        print("[DATA] processed (tokenized) data is dumped at ", to_fn)

def dump_vocab(tokenizer, to_fn):
    with open(to_fn, 'w', encoding='utf-8') as f:
        for k, v in tokenizer.vocab.items():
            print(f"{k}\t{v}", file=f)
        print("[VOCAB] file is dumped at ", to_fn)


def prepare_vocabs(src_tokenizer, tar_tokenizer, to_fns):

    info = {
        'source': {
            'pad' : src_tokenizer.pad_token_id,
            'unk' : src_tokenizer.unk_token_id
            },

        'target': {
            # note that, in case of GPT2, bos and eos are same
            # also, in decoder, 
            #   - we don't have pad id since we will handle pad-output symbol as -100 in cross-entropy
            #   - we don't have unknown symbol in generating process
            'bos' : tar_tokenizer.bos_token_id,
            'eos' : tar_tokenizer.eos_token_id   
            }
        }

    import json
    with open(to_fns['vocab_info'], 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        print("[VOCAB-INFO] file is dumped at ", to_fns['vocab_info'])


    dump_vocab(src_tokenizer, to_fns['source_vocab'])
    dump_vocab(tar_tokenizer, to_fns['target_vocab'])


if __name__ == '__main__':
    ## prepare data
    SOURCE_PRETRAINED_MODEL_NAME = 'klue/bert-base'  # source : Korean
    TARGET_PRETRAINED_MODEL_NAME = 'gpt2'            # target : English
    

    ## prepare tokenizers
    from transformers import BertTokenizerFast
    from transformers import GPT2TokenizerFast

    src_tokenizer = BertTokenizerFast.from_pretrained(SOURCE_PRETRAINED_MODEL_NAME)
    tar_tokenizer = GPT2TokenizerFast.from_pretrained(TARGET_PRETRAINED_MODEL_NAME)

    prepare_data(in_fns, src_tokenizer, tar_tokenizer)
    
    ## prepare vocabs
    prepare_vocabs(src_tokenizer, tar_tokenizer, to_fns)


