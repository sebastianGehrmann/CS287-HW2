#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

from collections import Counter

def construct_embeddings(embedding_file, num_words = 100000):
    emb_dict = {}
    with open(embedding_file, "r") as f:
        for i, line in enumerate(f):
            if i > num_words:
                break
            else:
                cline = line.split()
                emb_dict[transform_word(cline[0])] = np.array(cline[1:], dtype=np.float32)
    return emb_dict

def get_cap_feature(word):
    '''
    1: padding
    2: lower
    3: capitalized
    4: mixed
    5: all caps 
    '''
    cap_feature = 1
    if word.islower():
        cap_feature = 2
    elif word.isupper():
        cap_feature = 5
    elif word[0].isupper():
        cap_feature = 3
    else:
        cap_feature = 4
    return cap_feature

def transform_word(word):
    lw = word.lower()
    #can skip this -piazza id 56
    #lw = re.sub('\d+', 'NUMBER', lw)
    return lw

def get_vocab(file_list, embeddings, num_words=10000):
    word_to_idx = {}
    idx_to_embedding = []
    idx_to_embedding.append(np.random.randn(50))
    idx_to_embedding.append(np.random.randn(50))
    # Start at 3 (1 is PADDING, 2 is RARE)
    idx = 3
    for filename in file_list:
        if filename:
            with open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    cline = line.split()
                    if cline:
                        cword = transform_word(cline[2])
                        if cword not in word_to_idx and cword in embeddings:
                            word_to_idx[cword] = idx
                            idx_to_embedding.append(embeddings[cword])
                            idx += 1
    print(idx, "words have been counted")


    return word_to_idx, np.array(idx_to_embedding, dtype=np.float32)

def convert_data(data_name, word_to_idx, tag_to_idx, dwin):
    '''
    convert data to word indices for the window
    '''
    features = []
    capitalization = []
    lbl = []

    c = 0
    with open(data_name, 'r', encoding="latin-1") as f:
        sent = [1]*int((dwin-dwin%2)/2)
        cap = [1]*int((dwin-dwin%2)/2)
        for i, line in enumerate(f):
            # if i >= 5000:
            #     break
            cline = line.split()
            if len(sent) >= dwin:
                features.append(sent)
                capitalization.append(cap)
                sent = sent[1:]
                cap = cap[1:]

            if cline:
                #append features
                cword = transform_word(cline[2])
                if cword in word_to_idx.keys():
                    sent.append(word_to_idx[cword])

                else:
                    sent.append(2)    
                cap.append(get_cap_feature(cline[2]))
                try:
                    lbl.append(tag_to_idx[cline[3]])
                except:
                    pass
            else:
                #close sentence and reinitialize sent
                for i in range(int((dwin-dwin%2)/2)):
                    sent.append(1)
                    cap.append(1)
                    #for if it was already of the right size
                    if len(sent) > dwin:
                        sent = sent[1:]
                        cap = cap[1:]
                        features.append(sent)
                        capitalization.append(cap)
                    #for short sentences
                    if len(sent) == dwin:
                        features.append(sent)
                        capitalization.append(cap)
                        sent = sent[1:]
                        cap = cap[1:]
                sent = [1]*int((dwin-dwin%2)/2)
                cap = [1]*int((dwin-dwin%2)/2)
        #append last features
        for i in range(int((dwin-dwin%2)/2)):
            sent.append(1)
            cap.append(1)
            #for if it was already of the right size
            if len(sent) > dwin:
                sent = sent[1:]
                cap = cap[1:]
                features.append(sent)
                capitalization.append(cap)
            #for short sentences
            if len(sent) == dwin:
                features.append(sent)
                capitalization.append(cap)
                sent = sent[1:]
                cap = cap[1:]
        
    print(len(lbl), "size of y")
    print(len(capitalization), "size of cap vector")
    print(len(features), "size of features")
    # print()
    # print(features)
    # print()
    # print(lbl)
    return np.array(features, dtype=np.int32), np.array(capitalization, dtype=np.int32), np.array(lbl, dtype=np.int32)

        



def get_feature_map(tag_dict):
    tag_to_idx = {}
    with open(tag_dict, "r") as f:
        for line in f:
            cline = line.split()
            tag_to_idx[cline[0]] = cline[1]
    return tag_to_idx

FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict",
                      "data/glove.6B.50d.txt")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str, default="PTB")
    parser.add_argument('dwin', help="Size of Window",
                        type=int, default=5)
    parser.add_argument('size_vocab', help="Max size of Vocabulary",
                        type=int, default=100000)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = args.dwin
    vocab_size = args.size_vocab

    train, valid, test, tag_dict, emb = FILE_PATHS[dataset]
    word_emb = construct_embeddings(emb, vocab_size)

    word_to_idx, idx_to_embedding = get_vocab([train,valid,test], word_emb, vocab_size)
    tag_to_idx = get_feature_map(tag_dict)

    train_input, train_cap, train_output = convert_data(train, word_to_idx, tag_to_idx, dwin)
    if valid:
            valid_input, valid_cap, valid_output = convert_data(valid, word_to_idx, tag_to_idx, dwin)

    if test:
            test_input, test_cap, _ = convert_data(test, word_to_idx, tag_to_idx, dwin)

    V = len(word_to_idx) + 2
    print('Vocab size:', V)

    C = np.max(train_output)

    print(len(idx_to_embedding), "Embeddings")
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_cap'] = train_cap
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_cap'] = valid_cap
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
            f['test_cap'] = test_cap
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['dwin'] = np.array([dwin], dtype=np.int32)
        f['embeddings'] = idx_to_embedding


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
