from keras.layers import Dense, Input, RepeatVector, Lambda, Permute, Multiply, Concatenate
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.optimizers import RMSprop, Adam

from keras import backend as K
from keras.layers.wrappers import TimeDistributed

class _Attention(object):
    def __init__(self, ques_length, answer_length, nr_hidden, dropout=0.0, L2=0.0, activation='relu'):
        self.ques_length = ques_length
        self.answer_length = answer_length
    def __call__(self, sent1, sent2, reverse = False):
        def _outer(AB):
            att_ji = K.batch_dot(AB[1], K.permute_dimensions(AB[0], (0, 2, 1)))
            return K.permute_dimensions(att_ji,(0, 2, 1))
        if reverse:
            return Lambda(_outer,
                output_shape=(self.answer_length, self.ques_length))([sent2, sent1])
        else:
            return Lambda(_outer,
                output_shape=(self.ques_length, self.answer_length))([sent1, sent2])

class _SoftAlignment(object):
    def __init__(self, nr_hidden):
        # self.max_length = max_length
        self.nr_hidden = nr_hidden

    def __call__(self, sentence, attention, ques_len, max_length,  transpose=False):
        def _normalize_attention(attmat):
            att = attmat[0]
            mat = attmat[1]
            ques_len = attmat[2]
            if transpose:
                att = K.permute_dimensions(att,(0, 2, 1))
            # 3d softmax
            e = K.exp(att - K.max(att, axis=-1, keepdims=True))
            g = e * ques_len
            s = K.sum(g, axis=-1, keepdims=True) + K.epsilon()
            sm_att = g / s
            return K.batch_dot(sm_att, mat)
        return Lambda(_normalize_attention,
                      output_shape=(max_length, self.nr_hidden))([attention, sentence, ques_len]) # Shape: (i, n)


def DynamicClipAttention(model_param, elmo_embedding):
    hidden_dim = model_param["hidden_dim"]
    question = Input(shape=(1, ), dtype="string", name='question_base_inner')

    question_len = Input(shape=(model_param["enc_timesteps"],), dtype='float32', name='question_len')
    answer_len = Input(shape=(model_param["dec_timesteps"],), dtype='float32', name='answer_len')

    answer = Input(shape=(1, ), dtype="string", name='answer_good_base_inner') 

    embedding_layer_ques = Lambda(elmo_embedding)
    embedding_layer_ans = Lambda(elmo_embedding)

    question_emb = embedding_layer_ques(question)
    answer_emb = embedding_layer_ans(answer)


    ques_filter_repeat_len = RepeatVector(model_param["dec_timesteps"])(question_len)
    ans_filter_repeat_len = RepeatVector(model_param["enc_timesteps"])(answer_len)

    ans_repeat_len = RepeatVector(model_param["hidden_dim"])(answer_len)
    ans_repear_vec = Permute((2,1))(ans_repeat_len)

    ques_repeat_len = RepeatVector(model_param["hidden_dim"])(question_len)
    ques_repear_vec = Permute((2,1))(ques_repeat_len)

    SigmoidDense = Dense(hidden_dim,activation="sigmoid")
    TanhDense = Dense(hidden_dim,activation="tanh")

    QueTimeSigmoidDense = TimeDistributed(SigmoidDense,name="que_time_s")
    QueTimeTanhDense = TimeDistributed(TanhDense,name="que_time_t")

    AnsTimeSigmoidDense = TimeDistributed(SigmoidDense,name="ans_time_s")
    AnsTimeTanhDense = TimeDistributed(TanhDense,name="ans_time_t")


    question_sig = QueTimeSigmoidDense(question_emb)
    question_tanh = QueTimeTanhDense(question_emb)

    question_proj = Multiply()([question_sig,question_tanh])

    answer_sig = AnsTimeSigmoidDense(answer_emb)
    answer_tanh = AnsTimeTanhDense(answer_emb)

    answer_proj = Multiply()([answer_sig,answer_tanh])

    Attend = _Attention(model_param["enc_timesteps"], model_param["dec_timesteps"] , hidden_dim, dropout=0.2)
    Align = _SoftAlignment( hidden_dim)

    ques_atten_metrics = Attend(question_proj,answer_proj)
    ans_atten_metrics = Attend(question_proj,answer_proj,reverse = True)


    answer_align = Align(question_proj,ques_atten_metrics,ques_filter_repeat_len,model_param["dec_timesteps"], transpose=True)
    question_align = Align(answer_proj,ans_atten_metrics,ans_filter_repeat_len,model_param["enc_timesteps"],transpose=True)

    ans_temp_sim_output = Multiply()([answer_proj,answer_align])
    ques_temp_sim_output = Multiply()([question_proj,question_align])

    ans_sim_output = Multiply()([ans_temp_sim_output,ans_repear_vec])
    ques_sim_output = Multiply()([ques_temp_sim_output,ques_repear_vec])

    cnns = [Conv1D(kernel_size=filter_length,
                      filters=hidden_dim,
                      activation='relu',
                      padding='same') for filter_length in [1,2,3,4,5]]

    cnn_feature = Concatenate()([cnn(ans_sim_output) for cnn in cnns])
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    cnn_pool = maxpool(cnn_feature)

    OutputDense = Dense(hidden_dim,activation="relu")
    feature = OutputDense(cnn_pool)

    cnns1 = [Conv1D(kernel_size=filter_length,
                      filters=hidden_dim,
                      activation='relu',
                      padding='same') for filter_length in [1,2,3,4,5]]

    cnn1_feature = Concatenate()([cnn(ques_sim_output) for cnn in cnns1])
    cnn1_pool = maxpool(cnn1_feature)

    OutputDense1 = Dense(hidden_dim,activation="relu")

    feature1 = OutputDense1(cnn1_pool)

    feature_total = Concatenate()([feature,feature1],)

    FinalDense = Dense(hidden_dim, activation="relu")
    feature_all = FinalDense(feature_total)

    ScoreDense = Dense(1)#, activation="relu")
    score = ScoreDense(feature_all)

    basic_model = Model(inputs=[question,answer,question_len,answer_len],outputs=[score])

    questions = Input(
        shape=(1,), dtype='string', name='question_base')

    question_lens = Input(shape=(model_param["enc_timesteps"],), dtype='float32', name='question_len')

    good_answer = Input(
        shape=(1,), dtype='string', name='answer_base')
    answers = Input(
        shape=(model_param["random_size"],), dtype='string', name='answer_bad_base')

    answers_length = Input(shape=(model_param["random_size"],model_param["dec_timesteps"],), dtype='float32', name='answers_length')
    good_answer_length = Input(shape=(model_param["dec_timesteps"],),dtype='float32', name='good_answer_len')

    good_similarity = basic_model([questions, good_answer, question_lens,good_answer_length])

    sim_list = []
    for i in range(model_param["random_size"]):
        convert_layer = Lambda(lambda x:x[:,i],output_shape=(model_param["dec_timesteps"],))
        temp_tensor = convert_layer(answers)
        temp_length = convert_layer(answers_length)
        temp_sim = basic_model([questions,temp_tensor,question_lens,temp_length])
        sim_list.append(temp_sim)
    total_sim = Concatenate()(sim_list)
    total_prob = Lambda(lambda x: K.log(K.softmax(x + K.epsilon())),
                        output_shape = (model_param["random_size"], ))(total_sim)


    prediction_model = Model(
        inputs=[questions, good_answer,question_lens,good_answer_length],
        outputs=good_similarity, name='prediction_model')

    prediction_model.compile(
        loss="binary_crossentropy",
        optimizer = RMSprop(lr=model_param["lr"],
                       clipnorm=1.0,
                       clipvalue=0.5))

    training_model = Model(
        inputs=[questions, answers,question_lens,answers_length], outputs=total_prob, name='training_model')

    training_model.compile(
        loss=lambda y_true,y_pred: K.mean(y_true*(K.log(K.clip(y_true,0.00001,1)) - y_pred )) ,
        optimizer=Adam(lr=model_param["lr"],
                   beta_1=0.9,
                   beta_2=0.999,
                   clipvalue=0.5,
                   clipnorm=1.0,
                   epsilon=1e-04
                   ))
    return training_model, prediction_model