"""
    NLP 101 > N2N Classification (Named Entity Recognition) - Preprocessor (Vocab)

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""

in_fns = {
                'train': './data/ner/train.tsv',
                'valid': './data/ner/valid.tsv',
                'test':  './data/ner/test.tsv',
        }
to_fns = {
            'label_vocab': './data/ner/label.vocab',
            'token_vocab': './data/ner/token.vocab',
            'vocab_info': './data/ner/vocab.info.json',
         }


## prepare global label vocab
def load_data(fn):
    # output as pandas
    import pandas as pd 

    chars = []
    labels = []

    a_sent = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            if line == '':
                # end of a sentence
                if len(a_sent) >= 1:
                    chars.append( [x[0] for x in a_sent] )
                    labels.append( [x[1] for x in a_sent] )
                    a_sent = []
                continue
            else:
                fields = line.split('\t')
                char, label = fields[0], fields[-1]
                a_sent.append( (char, label) ) # character level

    data = {
                'chars': chars,
                'labels': labels
    }
    ## pandas
    df = pd.DataFrame(data)
    return df 




def get_label_vocab(fn, to_fn):
    df = load_data(fn)
    
    labels = []
    for a_sent_label in df.labels.tolist():
        for label in a_sent_label:
            labels.append(label)

    labels = sorted( list( set( labels ) ) )
    labels = labels # for handling padding labels
    label_vocab = { key:idx for idx, key in enumerate(labels)}

    with open(to_fn, 'w', encoding='utf-8') as f:
        for label, idx in label_vocab.items():
            print(f"{label}\t{idx}", file=f)

    print("[DUMP] Label vocab. is dumped at ", to_fn)
    return label_vocab


def get_token_vocab(model_name, token_fn):   
    # token vocab
    # checkout : https://github.com/kiyoungkim1/LMkor
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    SEP_ID = tokenizer.sep_token_id

    with open(token_fn, 'w', encoding='utf-8') as f:
        for t, idx in tokenizer.vocab.items():
            print(f"{t}\t{idx}", file=f)
        print("[TOKEN-VOCAB] file is dumped at ", token_fn)

    info = (tokenizer.vocab, tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.unk_token_id )
    return info

def dump_vocabs(token_vocab_info, label_vocab, to_fn):
    token_vocab, pad_token_id, sep_token_id, unk_token_id = token_vocab_info

    info = {
        'input': {
            'pad' : pad_token_id,
            'sep' : sep_token_id,
            'unk' : unk_token_id
            },
        'output' : {
            'pad' : -100 # pytorch cross-entropy default
            }
        }

    import json
    with open(to_fn, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
        print("[VOCAB-INFO] file is dumped at ", to_fn)


if __name__ == '__main__':
    ## prepare all vocabs 
    # get token vocab 
    PRETRAINED_MODEL_NAME = 'klue/bert-base'
    token_vocab_info = get_token_vocab(PRETRAINED_MODEL_NAME, to_fns['token_vocab'])
    
    # get label vocab
    label_vocab = get_label_vocab(in_fns['train'], to_fns['label_vocab'])

    # dump meta info
    dump_vocabs(token_vocab_info, label_vocab, to_fns['vocab_info'])






