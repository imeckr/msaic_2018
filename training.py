import os
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from keras import backend as K

from tqdm import tqdm
from keras.metrics import categorical_accuracy
from dynamic_clip_attention import DynamicClipAttention

from utils import mean_reciprocal_rank, parallelize_dataframe, tokenize, specific_save_epoch, prepare_submission, ListDataGenerator


## To prevent training to fail (MacOS)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


PROC_DATA_PATH = "processed"

MODEL_NAME = "DYNAMIC_CLIP_ATTENTION_ELMO_WE"
MAX_LEN_ENCODING_QUERY = 15
MAX_LEN_ENCODING_PASSAGE = 70
MODEL_REPO = "model_weights"

MODEL_PATH = os.path.join(MODEL_REPO, MODEL_NAME)


train = pd.read_csv(os.path.join(PROC_DATA_PATH,"undersample_train.tsv"), sep= "\t")
val = pd.read_csv(os.path.join(PROC_DATA_PATH,"val.tsv"), sep= "\t")



model_param = dict(hidden_dim=100,
                   enc_timesteps=MAX_LEN_ENCODING_QUERY,
                   dec_timesteps=MAX_LEN_ENCODING_PASSAGE,
                   random_size=4,
                   lr=0.001
                  )

# Elmo embedding definition
elmo_model = hub.Module('https://tfhub.dev/google/elmo/2', trainable= True)
def ElmoEmbedding(x):
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)),
                      as_dict=True,
                      signature="default")["word_emb"]


# Preprocess the data for Elmo format
def preprocess(df):
    df["passage_text"], df["passage_mask"] = zip(*df.passage_text.apply(lambda x: tokenize(x, MAX_LEN_ENCODING_PASSAGE)))
    df["query"], df["query_mask"] = zip(*df["query"].apply(lambda x: tokenize(x, MAX_LEN_ENCODING_QUERY)))
    return df

train = parallelize_dataframe(train, preprocess)
val = parallelize_dataframe(val, preprocess)

undersampled_train = ListDataGenerator(train, batch_size=64)

## Initialize model
training_model, prediction_model = DynamicClipAttention(model_param, ElmoEmbedding)

## Initializing variable
sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())


# Variable to store best MRR
best_val_mrr = 0.0

# 3 epochs seems to give the optimum MRR on validation dataset
NUM_EPOCHS = 10

for epoch in range(0, NUM_EPOCHS):
    training_model.fit_generator(undersampled_train,
                        epochs=(epoch+1),
                        verbose=1,
                        class_weight=None,
                        initial_epoch=epoch)
    print("Done")
    try:
        val_preds = prediction_model.predict([val["query"].values,
                                      val["passage_text"].values,
                                      np.vstack(val.query_mask.values),
                                      np.vstack(val.passage_mask.values)],
                                      batch_size = 128,
                                      verbose = 1)
        val_mrr = mean_reciprocal_rank(val_preds, val.label)
        print("Validation mrr:{}".format(val_mrr))
        if val_mrr >  best_val_mrr:
            best_val_mrr = val_mrr
            specific_save_epoch(training_model, MODEL_PATH)
    except Exception as e:
        print(str(e))


# Predict on test dataset
test = pd.read_csv(os.path.join(DATASET_PATH,"test.tsv"), sep= "\t")
test = parallelize_dataframe(test, preprocess)
test_preds = prediction_model.predict([test["query"].values,
                                      test["passage_text"].values,
                                      np.vstack(test.query_mask.values),
                                      np.vstack(test.passage_mask.values)],
                                      batch_size = 256,
                                      verbose = 1)

prepare_submission(test_dataset, MODEL_NAME)