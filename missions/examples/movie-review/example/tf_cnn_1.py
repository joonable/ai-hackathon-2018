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
from torch.utils.data import DataLoader

import nsml
# import dataset_split
# import kor_char_parser_with_masking
from dataset_split import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML
from kor_char_parser_with_masking import get_voca_num, masking
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
        :return:ƒ
        """

        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        # output_prediction = model(preprocessed_data)
        # point = output_prediction.data.squeeze(dim=1).tolist()

        pred = sess.run(predictions, feed_dict = {textCNN.input_x: preprocessed_data, textCNN.dropout_keep_prob : 1.0})
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

class TextCNN:
    def __init__(self, config, clip_gradients=5.0):
        """init all hyperparameter here"""
        self.num_classes = 11
        self.batch_size = config.batch
        self.sequence_length = config.strmaxlen
        self.vocab_size = vocab_size
        self.embed_size = config.embedding
        self.hidden_size = config.embedding
        self.learning_rate = config.lr
        self.initializer = tf.keras.initializers.he_normal()
        self.global_step = tf.Variable(0, trainable = False)
        self.decay_steps = config.decay_steps
        self.decay_rate = config.decay_rate


        self.filter_sizes=[3, 4, 5] # it is a list of int. e.g. [3,4,5]
        self.num_filters=32
        self.num_filters_total=self.num_filters * len(self.filter_sizes) #how many filters totally.
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y = tf.placeholder(tf.int32, [None,],name="input_y")  # y:[None,num_classes]
        self.dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32) #training iteration
        self.tst=tf.placeholder(tf.bool)

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step=tf.Variable(0,trainable=False,name="Epoch_Step")
        self.epoch_increment=tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
        self.b1 = tf.Variable(tf.ones([self.embed_size]) / 10)
        self.b2 = tf.Variable(tf.ones([self.embed_size]) / 10)


        self.instantiate_weights()
        self.logits = self.inference() #[None, self.label_size]. main computation graph is here.
        self.possibility=tf.nn.sigmoid(self.logits)

        self.loss_val = self.loss()

        self.train_op = self.train()

        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions,tf.int32), self.input_y) #tf.argmax(self.logits, 1)-->[batch_size]
        self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy") # shape=()

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding"): # embedding matrix
            self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.W_projection = tf.get_variable("W_projection",shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer) #[embed_size,label_size]
            self.b_projection = tf.get_variable("b_projection",shape=[self.num_classes])       #[label_size] #ADD 2017.06.09

    def inference(self):
        """main computation graph here: 1.embedding-->2.CONV-RELU-MAX_POOLING-->3.linear classifier"""
        # 1.=====>get emebedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)#[None,sentence_length,embed_size]
        self.sentence_embeddings_expanded=tf.expand_dims(self.embedded_words,-1) #[None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv

        # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
        # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
        pooled_outputs = []
        for i,filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" %filter_size):
                # ====>a.create filter
                filter=tf.get_variable("filter-%s"%filter_size, [filter_size,self.embed_size,1,self.num_filters], initializer=self.initializer)
                # ====>b.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                #Conv.Input: given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
                #Conv.Returns: A `Tensor`. Has the same type as `input`.
                #         A 4-D tensor. The dimension order is determined by the value of `data_format`, see below for details.
                #1)each filter with conv2d's output a shape:[1,sequence_length-filter_size+1,1,1];2)*num_filters--->[1,sequence_length-filter_size+1,1,num_filters];3)*batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                #input data format:NHWC:[batch, height, width, channels];output:4-D
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1,1,1,1], padding="VALID",name="conv")
                #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.layers.batch_normalization(conv)
                # ====>c. apply nolinearity
                b=tf.get_variable("b-%s"%filter_size,[self.num_filters]) #ADD 2017-06-09
                h=tf.nn.relu(tf.nn.bias_add(conv,b),"relu") #shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                #       ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                #       strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                pooled=tf.nn.max_pool(h, ksize=[1,self.sequence_length-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID',name="pool")
                #shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
        #e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
        #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
        #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
        self.h_pool=tf.concat(pooled_outputs,3) #shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat=tf.reshape(self.h_pool,[-1,self.num_filters_total]) #shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        #4.=====>add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob) #[None,num_filters_total]

        #5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop,self.W_projection) + self.b_projection  #shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits


    def loss(self,l2_lambda=0.0001):#0.001
        with tf.name_scope("loss"):
            #input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            #output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            #sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            #print("1.sparse_softmax_cross_entropy_with_logits.losses:",losses) # shape=(?,)
            loss=tf.reduce_mean(losses)#print("2.loss.loss:", loss) #shape=()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss=loss+l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)

        # train_op = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_val, global_step = self.global_step)

        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        gvs = optimizer.compute_gradients(self.loss_val)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs)

        # train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)

        return train_op

#test started. toy task: given a sequence of data. compute it's label: sum of its previous element,itself and next element greater than a threshold, it's label is 1,otherwise 0.
#e.g. given inputs:[1,0,1,1,0]; outputs:[0,1,1,1,0].
#invoke test() below to test the model in this toy task.

def get_label_y(input_x):
    length=input_x.shape[0]
    input_y=np.zeros((input_x.shape))
    for i in range(length):
        element=input_x[i,:] #[5,]
        result=compute_single_label(element)
        input_y[i,:]=result
    return input_y

def compute_single_label(listt):
    result=[]
    length=len(listt)
    for i,e in enumerate(listt):
        previous=listt[i-1] if i>0 else 0
        current=listt[i]
        next=listt[i+1] if i<length-1 else 0
        summ=previous+current+next
        if summ>=2:
            summ=1
        else:
            summ=0
        result.append(summ)
    return result

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type = int, default = 1)
    args.add_argument('--epochs', type = int, default = 10)
    args.add_argument('--batch', type = int, default = 1600)
    args.add_argument('--strmaxlen', type = int, default = 128)
    args.add_argument('--embedding', type = int, default = 8)
    args.add_argument('--lr', type = float, default = 0.01)
    args.add_argument('--decay_steps', type = int, default = 32)
    args.add_argument('--decay_rate', type = int, default = 0.9)
    # args.add_argument('--strmaxlen', type = int, default = 8)
    # args.add_argument('--embedding', type = int, default = 2)

    config = args.parse_args()


    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    input_size = config.embedding * config.strmaxlen
    output_size = 1


    textCNN = TextCNN(config)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess = sess, config = config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope = locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset_train = MovieReviewDataset(DATASET_PATH, config.strmaxlen)
        masking(dataset_train.x_test)
        vocab_size = get_voca_num()

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
            val_loss = 0.0
            min_val_loss = 10000
            for i, (data, labels) in enumerate(train_loader):
                loss, predictions, _ \
                    = sess.run([textCNN.loss_val, textCNN.predictions, textCNN.train_op],
                               feed_dict = {textCNN.input_y:labels, textCNN.input_x:data, textCNN.dropout_keep_prob:0.8})
                if i % 100 == 0:
                    val_loss = sess.run(textCNN.loss_val,
                                        feed_dict = {textCNN.input_y:y_test, textCNN.input_x:x_test,
                                                     textCNN.dropout_keep_prob:1})
                    print('Batch : ', i + 1, '/', total_batch, ', MSE in this minibatch: '
                          , float(loss), 'val_loss : ', val_loss)

                avg_loss += float(loss)

            train_loss = float(avg_loss / total_batch)
            if val_loss < min_val_loss:
                min_val_loss = val_loss

            print('epoch:', epoch, ' train_loss:', train_loss, ' val_loss:', val_loss)

            # nsml.report(summary = True, scope = locals(), epoch = epoch, epoch_total = config.epochs,
            #             train__loss = train_loss, val__loss = val_loss, step = epoch)
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