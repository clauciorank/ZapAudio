from FlagEmbedding import BGEM3FlagModel
import pandas as pd
import numpy as np
import pickle


if __name__ == '__main__':

    df = pd.read_csv('transcribed_files.csv')

    # https://huggingface.co/BAAI/bge-m3

    model = BGEM3FlagModel('BAAI/bge-m3',
                        use_fp16=True)
    
    sentences_2 = df['text'].to_list()
    embeddings_2 = model.encode(sentences_2,return_dense=True, return_sparse=True,  return_colbert_vecs=True)

    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(f)

