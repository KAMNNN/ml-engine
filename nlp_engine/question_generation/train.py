import argparse

import bert_sqg
import conv_gpt

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train transformers for question generation')
    parser.add_argument('--bert_sqg', dest='bert_sqg', action='store_true')
    parser.add_argument('--gpt_qgen', dest='gpt_qgen', action='store_false')

    args = parser.parse_args()
    if(args.bert_sqg):
        bert_sqg.train()
    elif(args.gpt_qgen):
        conv_gpt.train()


