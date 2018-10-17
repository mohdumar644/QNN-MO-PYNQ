#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset
import tensorpack.dataflow.imgaug.deform
from tensorpack.tfutils.varreplace import remap_variables
import tensorflow as tf
import tensorpack.dataflow.imgaug.deform
import cv2
import argparse
import numpy as np
import os
import sys
import scipy.io
from dorefa import get_dorefa
np.set_printoptions(suppress=True)

BITW = 1
BITA = 2
BITG = 16

IMG_WIDTH = 28

class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_WIDTH], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or  'fc' in name or 'conv0' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def cabs(x):
            return tf.clip_by_value(x, 0.0, 1.0, name='cabs')

        def activate(x):
            return fa(cabs(x))


        image = tf.expand_dims(image, 3)

	with remap_variables(binarize_weight), \
                argscope(BatchNorm, momentum=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
 
                      .Conv2D('conv0', 64, 3, padding='SAME', use_bias=True)
                      .apply(fg)
                      #.BatchNorm('bn0')
                      .MaxPooling('pool0', 2, padding='SAME')                      
                      .apply(activate)
                      
                      .Conv2D('conv1', 64, 3, padding='SAME')
                      .apply(fg)
                      .BatchNorm('bn1')
                      .apply(activate)
                      .MaxPooling('pool1', 2, padding='SAME')
		       
                      .Conv2D('conv2', 64, 3, padding='VALID')
                      .apply(fg)
                      .BatchNorm('bn2')
                      .apply(activate)
 
                      .tf.nn.dropout(0.5 if is_training else 1.0)  
                      .FullyConnected('fc0', 512) 
                      .apply(cabs)
                      .FullyConnected('fc1', 10)())
                      
        tf.nn.softmax(logits, name='output')

        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))

        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        return total_cost

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
	return tf.train.AdamOptimizer(lr) 


def get_config():
    logger.auto_set_dir()

    # prepare dataset
    d1 = dataset.Mnist('train')
    data_train = RandomMixData([d1])
    data_test = dataset.Mnist('test') 

    augmentors = [
        #imgaug.Resize((IMG_WIDTH, IMG_WIDTH)),
        #imgaug.Brightness(30),
        #imgaug.Contrast((0.5, 1.5)),
        #tensorpack.dataflow.imgaug.deform.GaussianDeform(        [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],        (IMG_WIDTH,IMG_WIDTH), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchDataZMQ(data_train, 5)

    augmentors = [imgaug.Resize((IMG_WIDTH, IMG_WIDTH))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)

    return TrainConfig(
        data=QueueInput(data_train),
        callbacks=[
            ModelSaver(),
            InferenceRunner(data_test,
                            [ScalarStats('cost'), ClassificationError('wrong_tensor')])
        ],
        model=Model(),
        max_epoch=200,
    )
 

def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],

        output_names=['output']  
    )
    predictor = OfflinePredictor(pred_config) 
    
    transformers = imgaug.AugmentorList([
        imgaug.Resize((IMG_WIDTH, IMG_WIDTH))
    ] )
    
    for f in inputs:
        assert os.path.isfile(f), f
        img = cv2.imread(f,0).astype('float32') 
        img = img / 255.0
        img = img[np.newaxis,:,:] 
        assert img is not None
        outputs_all = predictor(img)  
        outputs = outputs_all[0]
        prob = outputs[0]
        ps = np.exp(prob) / np.sum(np.exp(prob), axis=0) 
        predict= np.argmax(prob)
        ret = prob.argsort()[-10:][::-1]
       
        print('Predicted Class: {0}'.format(ret[0]))
        

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
                        default='1,2,16')

    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')

    correct = 0
    args = parser.parse_args()
    if args.run:
        run_image(Model(), SaverRestore('train_log/mnist_dorefa/checkpoint'), args.run)
        sys.exit()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    #config.session_init = SaverRestore('train_log/mnist_dorefa/checkpoint')
    launch_train_with_config(config, SimpleTrainer())
