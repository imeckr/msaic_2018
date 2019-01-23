import pandas as pd
import numpy as np
import pickle
import os

from tqdm import tqdm

# Source file paths
inputFileName = "data/Data.tsv"   
testFileName = "data/eval1_unlabelled.tsv"

# Processed file paths
PROC_DATH_PATH = "processed"

# Set seed
np.random.seed(42)

# Column names
train_columns =  ["query_id", "query", "passage_text", "label", "passage_id"]
test_columns =  ["query_id", "query", "passage_text","passage_id"]

# Reading data
print("Loading Datasets...")
train = pd.read_csv(inputFileName, sep="\t", header = None, names = train_columns)
test = pd.read_csv(testFileName, sep="\t", header = None, names = test_columns)
print("Dataset Loaded...")

#Number of training and testing examples
print("Train: {}, Test: {}".format(train.shape, test.shape))

#Sorting just to order
train = train.sort_values(["query_id","passage_id"]).reset_index(drop=True)
test = test.sort_values(["query_id","passage_id"]).reset_index(drop=True)


#Creating a local validation set
print("Splitting training dataset in to training and validation...")
all_training_query_ids = train.query_id.unique()
train_q_ids = np.random.choice(all_training_query_ids,
                               size= int(0.9 * len(all_training_query_ids)),
                               replace=False,
                               p=None)

#Split the entire training dataset further to training and validation set
train_dataset = train[train.query_id.isin(train_q_ids)].reset_index().drop("index", axis=1)
val_dataset = train[~train.query_id.isin(train_q_ids)].reset_index().drop("index", axis=1)

#Number of training examples
print("Train: {}, Validation: {}".format(train_dataset.shape, val_dataset.shape))


print("Undersampling trainin dataset from 9:1(Negative:Positve) to 3:1...")
# Undersample data from 9:1(Negative:Positve) to 3:1
pos_indices = np.array(train_dataset[train_dataset.label==1].index)
neg_indices = np.array(train_dataset[train_dataset.label!=1].index)
print("Positive examples:{}, Negative examples:{}".format(pos_indices.shape, neg_indices.shape))

TRAIN_QUERIES_NUMBER = int(train_dataset.query_id.shape[0]/10)
neg_indices_reshaped = (neg_indices.reshape(TRAIN_QUERIES_NUMBER, 9))

neg_indices_shuffled = np.zeros(neg_indices_reshaped.shape, dtype = "int32")
for i,row in enumerate(tqdm(neg_indices_reshaped)):
    neg_indices_shuffled[i,:] = np.random.permutation(neg_indices_reshaped[i,:])

undersample_neg_indices = neg_indices_shuffled[:,:3]
pos_indices = pos_indices.reshape((len(pos_indices),1))

undersample_train = np.hstack([undersample_neg_indices, pos_indices]).flatten()
train_undersample = train_dataset.loc[undersample_train,:].reset_index(drop = True)

## Saving undersample dataset, validation and test dataset
print("Saving dataset splits...")
train_undersample.to_csv(os.path.join(PROC_DATH_PATH,"undersample_train.tsv"), sep= "\t", index=False)
val_dataset.to_csv(os.path.join(PROC_DATH_PATH,"val.tsv"), sep= "\t", index=False)
test.to_csv(os.path.join(PROC_DATH_PATH,"test.tsv"), sep= "\t", index=False)
print("Done")

# The below part is needed in case you want to train the model with embeddings other than Elmo like Glove or Word2Vec
# Parameters
# VOCABULARY_SIZE = 300000
# MAX_LEN_ENCODING_QUERY = 15
# MAX_LEN_ENCODING_PASSAGE = 70
# EMBEDDING_DIMS = 300
# Creating a tokenizer to encode the data for neural network training
# filters= '!"#$%&()*+,-./:;<=>?@[\]^_`¨´{|}…’~\'£°“·”‘'
# tk = text.Tokenizer(num_words= VOCABULARY_SIZE, lower= True, filters= filters )
# tk.fit_on_texts(list(train_dataset["query"].values) +
#                 list(train_dataset["passage_text"].values))


# train_query_input = tk.texts_to_sequences(train_undersample["query"].values)
# train_query_input = sequence.pad_sequences(train_query_input,
#                                            padding = "post",
#                                            maxlen=MAX_LEN_ENCODING_QUERY)

# train_pt_input = tk.texts_to_sequences(train_undersample["passage_text"].values)
# train_pt_input = sequence.pad_sequences(train_pt_input,
#                                         padding = "post",
#                                         maxlen=MAX_LEN_ENCODING_PASSAGE)

# train_y = train_undersample.label.values
# pickle.dump((train_undersample["query_id"].values,
#              train_undersample["passage_id"].values,
#              train_query_input,
#              train_pt_input,
#              train_y),
#             open(os.path.join(PROC_DATH_PATH, "tokenized/undersample_train.p"), "wb"))


# val_query_input = tk.texts_to_sequences(val_dataset["query"].values)
# val_query_input = sequence.pad_sequences(val_query_input,
#                                          padding = "post",
#                                          maxlen=MAX_LEN_ENCODING_QUERY)

# val_pt_input = tk.texts_to_sequences(val_dataset["passage_text"].values.astype(str))
# val_pt_input = sequence.pad_sequences(val_pt_input,
#                                       padding = "post",
#                                       maxlen=MAX_LEN_ENCODING_PASSAGE)

# val_y = val_dataset.label.values

# pickle.dump((val_dataset["query_id"].values,
#              val_dataset["passage_id"].values,
#              val_query_input,
#              val_pt_input,
#              val_y), open(os.path.join(PROC_DATH_PATH, "tokenized/val.p"), "wb"))


# test_query_input = tk.texts_to_sequences(test["query"].values)
# test_query_input = sequence.pad_sequences(test_query_input,
#                                          padding = "post",
#                                          maxlen=MAX_LEN_ENCODING_QUERY)

# test_pt_input = tk.texts_to_sequences(test["passage_text"].values.astype(str))
# test_pt_input = sequence.pad_sequences(test_pt_input,
#                                       padding = "post",
#                                       maxlen=MAX_LEN_ENCODING_PASSAGE)

# pickle.dump((test["query_id"].values,
#              test["passage_id"].values,
#              test_query_input,
#              test_pt_input), open(os.path.join(PROC_DATH_PATH, "tokenized/test.p"), "wb"))

# pickle.dump(dict(VOCABULARY_SIZE = VOCABULARY_SIZE,
# MAX_LEN_ENCODING_QUERY = MAX_LEN_ENCODING_QUERY,
# MAX_LEN_ENCODING_PASSAGE = MAX_LEN_ENCODING_PASSAGE,
# EMBEDDING_DIMS = EMBEDDING_DIMS),
#             open(os.path.join(PROC_DATH_PATH, "tokenized/config.p"), "wb"))

# pickle.dump(tk, open(os.path.join(PROC_DATH_PATH, "tokenized/tokenizer.p"), "wb"))
