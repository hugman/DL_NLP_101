"""
    NLP 101 > N2M Translation (Neural Machine Translation) - Tester

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmai.com)
    All rights reserved. (2022)
"""


# In this code, we will implement
#   - load model 
#   - simple decoding using top-k sampling 
#
#   - If you want to use more-advanced generation such as beam, 
#   - checkout https://huggingface.co/docs/transformers/internal/generation_utils

from types import prepare_class
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from practice_3_n2m_1_ner_trainer import NMT_DataModule, NeuralMachineTranslation
from commons.base import top_k_logits

@torch.no_grad()
def sample_a_sequence(model, enc_output, src_mask, dec_input, steps, temperature=1.0, sample=False, top_k=None, end_symbol_id=None):
    ## note that 
    ## this method is just for showing how to generate from the decoder
    ## using probability distribution in educational purpose

    ## for speed, diversity and many other reasons
    ## you must use other well-designed generation utilities 
    ## for serious applications, you must check other tools
    ## ex) https://huggingface.co/docs/transformers/internal/generation_utils
    
    model.eval()
    for k in range(steps):
        logits, seq_hidden_states, layers_attn_scores_1, layers_attn_scores_2 = model.forward_decoder(enc_output, src_mask, dec_input)
        # pluck the logits at the final step and scale by temperature

        logits = logits[:, -1, :] / temperature
        
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        if end_symbol_id != None:
            if end_symbol_id == ix:
                break 

        # append to the sequence and continue
        dec_input = torch.cat((dec_input, ix), dim=1)
        

    return dec_input

if __name__ == '__main__':
    import torchmetrics
    
    # ------------
    # data
    # ------------
    dm = NMT_DataModule(batch_size=1)  ## for just easy implemenation (in real, you need to set bigger value)
    dm.prepare_data()
    dm.setup()
    x = iter(dm.test_dataloader()).next() # <for testing 

    import pandas as pd
    df = pd.read_pickle('./data/translation/test.processed.pkl')

    # ------------
    # Model
    # ------------
    model_fn =  "./result/nmt/best_model.ckpt"
    model = NeuralMachineTranslation.load_from_checkpoint(model_fn).to('cuda:0')

    decoder_bos_id, decoder_eos_id = dm.vocab_info['target']['bos'], dm.vocab_info['target']['eos']
    r_dec_vocab = { v:k for k,v in dm.target_vocab.items() }

    from transformers import GPT2TokenizerFast
    TARGET_PRETRAINED_MODEL_NAME = 'gpt2'            # target : English
    tar_tokenizer = GPT2TokenizerFast.from_pretrained(TARGET_PRETRAINED_MODEL_NAME)

    ## ------------
    ## testing
    ## ------------
    import tqdm
    gen_texts = []
    bleu_scores = []

    to_f = open('./result/nmt/test.result.txt', 'w', encoding='utf-8')

    for ex_idx, batch in tqdm.tqdm(enumerate(dm.test_dataloader()), total=len(dm.test_dataloader())):
        #src_token_ids, src_mask, tar_token_ids, tar_output_ids = batch 
        src_token_ids, src_mask, _, _ = batch 
        src_token_ids = src_token_ids.to(model.device)
        src_mask = src_mask.to(model.device)

        # A,B,C --> 1,2,3,4,5
        # y = pred( input = [A, B, C, D] )

        # src_token_ids : [1, enc_seq_len]  # 1 for batch size
       
        # ------------
        # Encoding  (one time)
        # ------------
        enc_output = model.forward_encoder(src_token_ids, src_mask)

        # ------------
        # Decoding  (many time)
        # ------------
        # ex) <s> 1, 2, 3, 4, 5, </s>
        #
        # 1st : <s>        --> 1
        # 2nd : <s>, 1     --> 1, 2
        # 3rd : <s>, 1, 2, --> 1, 2, 3
        dec_input = torch.tensor([decoder_bos_id]).to(model.device)
        dec_input = dec_input.unsqueeze(dim=0) # [B, ~]
        a_generated_seq_ids = sample_a_sequence(model, enc_output, src_mask, dec_input, dm.dec_max_seq_len, temperature=1.0, sample=False, top_k=1, end_symbol_id=decoder_eos_id)

        a_generated_seq_ids = a_generated_seq_ids[0]  # since we have just one example
        a_generated_seq_ids = a_generated_seq_ids[1:] # remove <bos>
        a_generated_text    = tar_tokenizer.decode(a_generated_seq_ids)
        
        ## sentence to sentence bleu score
        translate_corpus = [a_generated_text]
        reference_corpus = [ [df.iloc[ex_idx].target] ]
        bleu_score = torchmetrics.functional.sacre_bleu_score(reference_corpus, translate_corpus)

        gen_texts.append(a_generated_text)
        bleu_scores.append(bleu_score.item())

        # dump 
        input_text = df.iloc[ex_idx].source
        reference_text = df.iloc[ex_idx].target
        predicted_text = a_generated_text
        print(f"SOURCE    : {input_text}", file=to_f)
        print(f"REFERENCE : {reference_text}", file=to_f)
        print(f"PREDICTED : {predicted_text}", file=to_f)
        print(f"-"*100, file=to_f)
        to_f.flush()

    to_f.close()


    df.translated = gen_texts
    df.bleu_scores = bleu_scores


    ## dump result
    df.to_pickle('./result/nmt/test.result.df.pkl')

    ## the BLEU will be araound 0.20 which is not so high-value
    ## this tutorial is just for showing how to build-up NMT using transformers with small sample data

