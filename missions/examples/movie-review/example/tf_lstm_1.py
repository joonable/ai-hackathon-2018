# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import nsml
from kor_char_parser_with_masking import get_voca_num, masking
from dataset_split import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

import tensorflow as tf

def bind_model(sess, config):

    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        os.makedirs(dir_name, exist_ok = True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """

        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        # output_prediction = model(preprocessed_data)
        # point = output_prediction.data.squeeze(dim=1).tolist()

        pred = sess.run(predictions, feed_dict = {textRNN.input_x: preprocessed_data, textRNN.dropout_keep_prob : 1.0})
        # pred = pred + 1
        # point = pred.data.squeeze(dim = 1).tolist()
        # print(pred)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다.
        # 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(pred)), pred))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def collate_fn(data: list):
    review = []
    label = []
    for datum in data:
        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)

class TextRNN:
    def __init__(self, config, vocab_size):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = 11
        self.batch_size = config.batch
        self.sequence_length=config.strmaxlen
        self.vocab_size=vocab_size
        self.embed_size=config.embedding
        self.hidden_size=config.embedding
        self.learning_rate=config.lr
        self.initializer=tf.contrib.layers.xavier_initializer()
        self.global_step = tf.Variable(0, trainable=False)
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32,[None], name="input_y")  # y [None,num_classes]
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")

        # self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        # self.epoch_increment=tf.assign(self.epoch_step, tf.add(self.epoch_step,tf.constant(1)))

        self.instantiate_weights()
        self.logits = self.inference()
        #[None, self.label_size]. main computation graph is here.

        self.loss_val = self.loss()  # -->self.loss_nce()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, axis = 1, name = "predictions")  # shape:[None,]
        # tf.argmax(self.logits, 1)-->[batch_size]

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            # self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
            #                                  initializer = self.initializer)
            self.Embedding = tf.get_variable('char_embedding', [self.vocab_size, config.embedding])

            #[vocab_size, embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)

            self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size*2, self.num_classes],
                                                initializer = self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes],
                                                initializer = self.initializer)       #[label_size]

    def inference(self):
        """main computation graph here: 1. embeddding layer, 2.Bi-LSTM layer, 3.concat, 4.FC layer 5.softmax """

        #1.get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        #shape:[None,sentence_length,embed_size]

        #2. Bi-lstm layer
        # define lstm cess : get lstm cell output
        lstm_fw_cell=tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size) #forward direction cell
        lstm_bw_cell=tf.contrib.rnn.LayerNormBasicLSTMCell(self.hidden_size) #backward direction cell
        lstm_fw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell,output_keep_prob=self.dropout_keep_prob)
        lstm_bw_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell,output_keep_prob=self.dropout_keep_prob)

        # bidirectional_dynamic_rnn: input: [batch_size, max_time, input_size] output: A tuple (outputs, output_states)
        # where:outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, dtype=tf.float32)
        #[batch_size, sequence_length, hidden_size] #creates a dynamic bidirectional recÍurrent neural network
        print("outputs:===>",outputs)
        #outputs:
        # (<tf.Tensor 'bidirectional_rnn/fw/fw/transpose:0' shape=(?, 5, 100) dtype=float32>,
        # <tf.Tensor 'ReverseV2:0' shape=(?, 5, 100) dtype=float32>))

        #3. concat output
        output_rnn = tf.concat(outputs, axis=2) #[batch_size, sequence_length, hidden_size*2]
        self.output_rnn_last=tf.reduce_mean(output_rnn, axis=1) #[batch_size, hidden_size*2]
        #output_rnn_last = output_rnn[:,-1,:]       #[batch_size,hidden_size*2]
        # print("output_rnn_last:", self.output_rnn_last) # <tf.Tensor 'strided_slice:0' shape=(?, 200) dtype=float32>

        #4. logits(use linear layer)
        with tf.name_scope("output"):
            #inputs: A `Tensor` of shape `[batch_size, dim]`.
            #The forward activations of the input network.
            logits = tf.matmul(self.output_rnn_last, self.W_projection) + self.b_projection
            # [batch_size,num_classes]
        return logits

    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            #input: `logits` and `labels` must have the same shape `[batch_size, num_classes]`
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            #sigmoid_cross_entropy_with_logits.
            # #losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        # learning_rate = tf.train.exponential_decay(
        #     self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        # train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
        #                                            self.decay_rate, staircase = True)
        # self.learning_rate_ = learning_rate
        train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_val, global_step = self.global_step)
        return train_op

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type = int, default = 1)
    args.add_argument('--epochs', type = int, default = 20)
    args.add_argument('--batch', type = int, default = 1600)
    args.add_argument('--strmaxlen', type = int, default = 128)
    args.add_argument('--embedding', type = int, default = 8)
    args.add_argument('--lr', type = float, default = 0.003)
    args.add_argument('--decay_steps', type = int, default = 32)
    args.add_argument('--decay_rate', type = int, default = 0.9)
    config = args.parse_args()

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    input_size = config.embedding * config.strmaxlen
    output_size = 1

    dataset_train = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
    masking(dataset_train.x_test)
    vocab_size = get_voca_num()

    textRNN = TextRNN(config, vocab_size = vocab_size)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess = sess, config = config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope = locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.

        y_test = dataset_train.y_test
        x_test = dataset_train.x_test

        train_loader = DataLoader(dataset = dataset_train,
                                  batch_size = config.batch,
                                  shuffle = True,
                                  collate_fn = collate_fn,
                                  num_workers = 2)

        total_batch = len(train_loader)

        min_val_loss = 100

        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):
                loss, predictions, _ \
                    = sess.run([textRNN.loss_val, textRNN.predictions, textRNN.train_op],
                               feed_dict = {textRNN.input_y:labels, textRNN.input_x:data, textRNN.dropout_keep_prob:0.8})

                if i % 100 == 0:
                    val_loss = sess.run(textRNN.loss_val,
                                        feed_dict = {textRNN.input_y:y_test, textRNN.input_x:x_test,
                                                     textRNN.dropout_keep_prob:1})
                    print('Batch : ', i + 1, '/', total_batch, ', MSE in this minibatch: '
                          , float(loss), 'val_loss : ', val_loss)

                avg_loss += float(loss)

            train_loss = float(avg_loss / total_batch)
            val_loss = sess.run(textRNN.loss_val,
                           feed_dict = {textRNN.input_y:y_test, textRNN.input_x:x_test, textRNN.dropout_keep_prob:1})
            if val_loss < min_val_loss:
                min_val_loss = val_loss

            print('epoch:', epoch, ' train_loss:', train_loss, ' val_loss:', val_loss)

            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch)

            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)



        print(res)