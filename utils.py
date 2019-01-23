import os
import h5py
import string
import zipfile

import pandas as pd
import numpy as np

from nltk import word_tokenize
from multiprocessing import Pool
from keras.utils import to_categorical, Sequence

num_partitions = 10 
num_cores = 4

class ListDataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, train, batch_size=128, shuffle=True, debug=False):
        'Initialization'
        
        self.unique_ques = len(np.unique(train.query_id.values))
        self.query_id = train.query_id.values.reshape((self.unique_ques,-1))
        self.query = train["query"].values.reshape((self.unique_ques,-1))[:,0]
        self.passage = train["passage_text"].values.reshape((self.unique_ques,-1))
        self.label = train.label.values.reshape((self.unique_ques,-1))
        self.query_len = np.vstack(train.query_mask.values).reshape((self.unique_ques,-1, MAX_LEN_ENCODING_QUERY))[:,0]
        self.passage_len = np.vstack(train.passage_mask.values).reshape((self.unique_ques,-1, MAX_LEN_ENCODING_PASSAGE))
        
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.debug = debug
        
        assert self.query_id.shape == self.label.shape
        
        self.on_epoch_end()

    def set_batch_size(batch_size):
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.unique_ques/ self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        if self.debug:
            X, y, _ = self.__data_generation(batch_indexes)
            return X, y, _
        else:
            X, y = self.__data_generation(batch_indexes)
            return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("Generating new batches")
        self.indexes = np.arange(len(self.query_id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        'Generates data containing batch_size samples'
        X = [self.query[batch_indexes],
             self.passage[batch_indexes],
             self.query_len[batch_indexes],
             self.passage_len[batch_indexes]]
        
        y = self.label[batch_indexes]
        if self.debug:
            return X, y, self.query_id[batch_indexes]
        return X, y


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


stopset = list(string.punctuation)

def tokenize(text, max_len):
    new_seq = []
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopset]
    for i in range(max_len):
        try:
            new_seq.append(tokens[i])
        except:
            new_seq.append("__PAD__")
    mask = np.array((np.array(new_seq)!= "__pad__"), dtype="int8")
    return " ".join(new_seq), mask

def specific_save_epoch(model,path):
    filename = '%s.h5' % (path)
    h5_file = h5py.File(filename,'w')
    weight = model.get_weights()
    for i in range(len(weight)):
        h5_file.create_dataset('weight'+str(i),data=weight[i])
    h5_file.close()

def specific_load_epoch(model,path):
    filename = '%s.h5' % (path)
    h5_file = h5py.File(filename,'r')
    weight = []
    for i in range(len(h5_file.keys())):
        weight.append(h5_file['weight'+str(i)][:])
    model.set_weights(weight)

def mean_reciprocal_rank(preds, labels, cases = 10):
    total_queries = int(len(labels)/cases)
    preds_ = preds.reshape((total_queries, cases))
    labels_ = labels.reshape((total_queries, cases))
    sorted_indices = (-preds_).argsort()
    rel = np.zeros((total_queries,cases))
    for i, index in enumerate(sorted_indices):
        rel[i,:] = labels_[i,:][index]
    return np.mean(1.0/(np.argmax(rel, axis = 1) + 1))

def prepare_submission(df, submission_name, base_path = ""):
    subm = (df.groupby("query_id")["score"]
            .apply(list).reset_index())
    subm[list(map(str,range(10)))] = subm["score"].apply(pd.Series)
    subm[["query_id"] + list(map(str,range(10))) ].to_csv("answer.tsv",
                                                          sep="\t",
                                                          index=False,
                                                          header=False)
    zipfile.ZipFile(os.path.join(base_path ,submission_name +'.zip'), mode='w').write("answer.tsv")
    os.remove("answer.tsv")  