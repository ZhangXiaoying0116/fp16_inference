import tensorflow as tf
import os
import time
import numpy as np
from tensorflow.python.framework import graph_util
os.environ["CUDA_VISIBLE_DEVICES"]="7"

test0 = True
if test0:
    def build_model(input1,input2):
        # build your model
        nets = tf.gather(input1, input2)
        nets = tf.add(nets,tf.constant(6.0))
        weight = tf.random_normal_initializer()
        nets = tf.matmul(nets, weight(shape=(2, 2), dtype="float32"))
        nets = tf.nn.softmax(nets)
        return nets
    input1 = tf.placeholder(tf.float32, shape=[2, 2])
    input2 = tf.placeholder(tf.int32, shape=[2])
    output = build_model(input1,input2)
    with tf.Session() as sess:
        result = sess.run([output], feed_dict={input1: [[100, 2], [3, 4]], input2: [1, 0]})
        print(result)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Softmax'])
        with tf.gfile.FastGFile('./saved_model.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

test1 = False
if test1:
    input1 = tf.placeholder(tf.float32, shape=[2, 2])
    input2 = tf.placeholder(tf.int32, shape=[2])
    inputs = tf.gather(input1, input2)
    inputs = tf.add(inputs,tf.constant(6.0))
    weight = tf.random_normal_initializer()
    inputs = tf.matmul(inputs, weight(shape=(2, 2), dtype="float32"))
    output = tf.nn.softmax(inputs)
    with tf.Session() as sess:
        result = sess.run([output], feed_dict={input1: [[100, 2], [3, 4]], input2: [1, 0]})
        print(result)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Softmax'])
        with tf.gfile.FastGFile('./1amp3_tensorflowModel.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

test2 = False
if test2:
    input1 = tf.placeholder(tf.float32, shape=[2, 2])
    inputs = tf.gather(input1, [1, 0])
    inputs = tf.add(inputs,tf.constant(6.0))
    inputs = tf.matmul(inputs, [[100.0, 2.22], [3.2, 4.1]])
    output = tf.nn.softmax(inputs)
    feed_dict = {input1: [[100, 2], [3, 4]]}
    with tf.Session() as sess:
        result = sess.run([output], feed_dict)
        print(result)
        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Softmax'])
        with tf.gfile.FastGFile('./2amp3_tensorflowModel.pb', mode='wb') as f:
            f.write(constant_graph.SerializeToString())

test3 = False
if test3:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1" 
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1"
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH'] ="./amp_log_test3_rn18/"
    graph_def = tf.GraphDef()
    bs = 2
    img = np.random.rand(bs, 224, 224, 3).astype(np.float32)
    with open("/home/devdata/xiaoying.zhang/AMP/resnet50_v1.5_imagenet2.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        config = tf.ConfigProto()
        # method2
        # config.graph_options.rewrite_options.auto_mixed_precision=1
        with tf.Session(config=config) as sess:
            input_image_tensor = sess.graph.get_tensor_by_name('IteratorGetNext:0')
            output_tensor = sess.graph.get_tensor_by_name('resnet_model/dense/BiasAdd:0')
            # compute fps
            time_start =time.time()
            for i in range(1000):
                output = sess.run([output_tensor], feed_dict={input_image_tensor: img})
            time_end =time.time()
            print('Inference fps:',1/((time_end-time_start)/(1000*bs)))

test4 = False
if test4:
    graph_def = tf.GraphDef()
    img = np.random.rand(2,2).astype(np.float32)
    with open("2amp3_tensorflowModel.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")
        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            input_image_tensor = sess.graph.get_tensor_by_name('Placeholder:0')
            output_tensor = sess.graph.get_tensor_by_name('Softmax:0')
            output = sess.run([output_tensor], feed_dict={input_image_tensor: img})
            print("output:",output)
