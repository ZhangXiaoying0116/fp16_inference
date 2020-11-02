from tensorflow.python.framework import graph_util
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('ckpt_314435/graph-0818-024944.meta') # meta for ckpt
    saver.restore(sess, "ckpt_314435_mixed/model-314435") # ckpt
    output_pb = graph_util.convert_variables_to_constants(
                        sess=sess,
                        input_graph_def=sess.graph_def,
                        output_node_names=['tower-pred-0/output/boxes', 'tower-pred-0/output/scores','tower-pred-0/output/labels'])
    tf.train.write_graph(output_pb, './ckpt_314435_mixed/', 'model-314435.pb', as_text=False)
    print("export pb files finish !")