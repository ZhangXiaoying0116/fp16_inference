# -*- coding: UTF-8 -*-
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2,tensor_shape_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.contrib import tensorrt as trt 
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
    parser = argparse.ArgumentParser(description='pb file inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--model-name', metavar='M', type=str, required=False, default='fasterrcnn' ,
                        help='model name', dest='modelname')
    parser.add_argument('-m', '--model-path', metavar='M', type=str, required=False, default='pbfiles/fasterrcnn-tensorpack.pb' ,
                        help='model path', dest='modelpath')
    parser.add_argument('-c1', "--count-op", help="count op numbers", action="store_true", dest='countop')
    parser.add_argument('-c2', "--ckpt2-pb", help="ckpt convert to pb", action="store_true", dest='ckpt2pb')
    parser.add_argument('-c3', "--check-ef32", help="check ef32 model", action="store_true", dest='checkef32')
    parser.add_argument('-c4', "--infer-f16", help="convert to f16 and do infer", action="store_true", dest='inferf16')
    parser.add_argument('-c5', "--infer-aotuamp", help="amp infer", action="store_true", dest='inferaotuamp')

    parser.add_argument('-s', '--save-path', metavar='S', type=str, required=False, default='./pbfiles' ,
                        help='save path', dest='savepath')
    parser.add_argument('-o', '--outpb-name', metavar='O', type=str, required=False, default='fasterrcnn-tensorpack-final' ,
                        help='outpb name', dest='outpbname')
    parser.add_argument('-t', '--target_type', metavar='T', type=str, required=False, default='fp16' ,
                        help='target type', dest='targettype')

    parser.add_argument('-in', '--input_names', type=list, required=False, default=['image:0'],
                        help='input names', dest='inputnames')
    parser.add_argument('-out', '--output_names', type=list, required=False, default=['tower-pred-0/output/boxes:0', 'tower-pred-0/output/scores:0','tower-pred-0/output/labels:0'],
                        help='outpb names', dest='outputnames')

    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()

args = get_args()

##############################################################################################################################
##功能描述：该文件涵盖pb文件的常用处理，针对所有pb模型自动化工具
##功能1：统计pb中算子的个数
##功能2：将ckpt的权重文件转换成pb文件
##功能3：检查ef32算子的转换，对于矩阵乘、卷积、可分离卷积是否完成相应的前向和反向的转换；前向input和filter插入转换，反向grad的输入插入转换
##功能4：手工转换fp32的原始模型至fp16，包含全部节点转换；对于无法转换的节点进行检查，并根据错误信息插入cast节点
##功能5：自动转换fp32模型至混合精度模型，不是全部节点转换，图结构可导出；功能5与功能4可进行同步分析，确定模型最合适的fp16模型转换方法
##############################################################################################################################

def load_graph(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        if model_path.endswith("pb"):
            with open(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
        else:
            with open(model_path, "r") as pf:
                text_format.Parse(pf.read(), graph_def)
        tf.import_graph_def(graph_def, name="")
        sess = tf.Session(graph=graph)
        return sess

## count op 
##############################################################################################################################
def count_op_numbers(model_path):
    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    ## Count the number of OP
    node_dict = {}

    for node in source_graph_def.node:
        ## Count the number of OP
        if node.op not in node_dict.keys():
            node_dict[node.op] = 1
        else:
            node_dict[node.op] += 1

    ## export op node list 
    print(node_dict)
    f = open(args.modelname + '_node_dict.csv','w')
    w = csv.DictWriter(f,node_dict.keys())
    w.writeheader()
    w.writerow(node_dict)
    f.close()
##############################################################################################################################

## ckpt2pb
##############################################################################################################################
def ckpt2pb(ckptmodel_path,ckptmodel_name,pbmodel_path,input_names=None,output_names=None):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(ckptmodel_path+ckptmodel_name+'.meta') # 'ckpt_314435/graph-0818-024944.meta'
        saver.restore(sess, ckptmodel_path+ckptmodel_name) # ckpt
        output_pb = graph_util.convert_variables_to_constants(
                            sess=sess,
                            input_graph_def=sess.graph_def,
                            output_node_names=output_names)
        # output_names: ['tower-pred-0/output/boxes', 'tower-pred-0/output/scores','tower-pred-0/output/labels']

        tf.train.write_graph(output_pb, pbmodel_path, ckptmodel_name+'.pb', as_text=False)
        print("export pb files finish !")
##############################################################################################################################

## count Matmul / Convolution / Separable Convolution whether fully inserted EF32 operator
##############################################################################################################################
def check_ef32_model(model_path):
    ## in ef32 docker:
    ## export tf114_ef32_forward to path
    import tf114_ef32_forward
    import tf114_ef32_backward
    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()

    checkef32bp = ''
    for node in source_graph_def.node:
        ## Check that the EF32 operator inserts are complete
        if checkef32bp in node.input and node.op != 'Convert2ef32bp':
            print(checkef32bp,":2-the bp grad was not converted to EF32")
            # assert(0)
        if node.op == "Conv2D":
            convinput = node.input[0]
            convfilter = node.input[1]
            if 'Convert2ef32fp' not in convinput:
                print(node.name,":0-the conv2d input was not converted to EF32 in fp")
                # assert(0)
            if 'Convert2ef32fp' not in convfilter:
                print(node.name,":1-the conv2d filter was not converted to EF32 in fp")
                # assert(0)
            checkef32bp = node.name
        elif node.op =='DepthwiseConv2dNative':
            depthwiseconvinput = node.input[0]
            depthwiseconvfilter = node.input[1]
            if 'Convert2ef32fp' not in depthwiseconvinput:
                print(node.name,":0-the depthwiseconv2d input was not converted to EF32 in fp")
                # assert(0)
            if 'Convert2ef32fp' not in depthwiseconvfilter:
                print(node.name,":1-the depthwiseconv2d filter was not converted to EF32 in fp")
                # assert(0)
            checkef32bp = node.name
        elif node.op =='MatMul':
            matmulinput = node.input[0]
            matmulfilter = node.input[1]
            if 'Convert2ef32fp' not in matmulinput:
                print(node.name,":0-the matmul input was not converted to EF32 in fp")
                # assert(0)
            if 'Convert2ef32fp' not in matmulfilter:
                print(node.name,":1-the matmul filter was not converted to EF32 in fp")
                # assert(0)
            checkef32bp = node.name
##############################################################################################################################

## fp16 inference
##############################################################################################################################
def rewrite_batch_norm_node_v2(node, graph_def, target_type='fp16'):
    """
    Rewrite FusedBatchNorm with FusedBatchNormV2 for reserve_space_1 and reserve_space_2 in FusedBatchNorm require float32 for 
    gradient calculation (See here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/fused-batch-norm)
    """
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    new_node = graph_def.node.add()
    new_node.op = "FusedBatchNormV2"
    new_node.name = node.name
    new_node.input.extend(node.input)
    """
    Value for attr 'U' of half is not in the list of allowed values: float; 
    NodeDef: {{node tower-pred-0/conv0/bn/FusedBatchNorm}}; 
    Op<name=FusedBatchNormV2; signature=x:T, scale:U, offset:U, mean:U, variance:U 
    -> y:T, batch_mean:U, batch_variance:U, reserve_space_1:U, reserve_space_2:U; 
    attr=T:type,allowed=[DT_HALF, DT_BFLOAT16, DT_FLOAT]; attr=U:type,allowed=[DT_FLOAT]; 
    attr=epsilon:float,default=0.0001; attr=data_format:string,default="NHWC",allowed=["NHWC", "NCHW"]; 
    attr=is_training:bool,default=true>
    """
    new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))

    for attr in list(node.attr.keys()):
        if attr == "T":
            node.attr[attr].type = dtype
        new_node.attr[attr].CopyFrom(node.attr[attr])
    print("rewrite fused_batch_norm done!")

def convert_graph_all2fp16(model_path, target_type='fp16', first_modify=True, input_names=None, output_names=None):
    if target_type == 'fp16':
        dtype = types_pb2.DT_HALF
    elif target_type == 'fp64':
        dtype = types_pb2.DT_DOUBLE
    else:
        dtype = types_pb2.DT_FLOAT
    source_sess = load_graph(model_path)
    source_graph_def = source_sess.graph.as_graph_def()
    target_graph_def = graph_pb2.GraphDef()
    target_graph_def.versions.CopyFrom(source_graph_def.versions)

    for node in source_graph_def.node:
        # fused batch norm node
        if node.op == "FusedBatchNorm":
            rewrite_batch_norm_node_v2(node, target_graph_def, target_type=target_type)
            continue
        # replicate node
        new_node = target_graph_def.node.add()
        new_node.op = node.op
        new_node.name = node.name
        new_node.input.extend(node.input)
        attrs = list(node.attr.keys())
        # keep batch norm params node
        if ("BatchNorm" in node.name) or ('batch_normalization' in node.name) \
                or ('bn/gamma' in node.name) or ('bn/beta' in node.name) or ('bn/mean' in node.name) or ('bn/variance' in node.name): ## make sure FusedBatchNormV2'inputs scale/offset/mean/variance will be U/float type
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp32
            if node.name in keep_fp32_node_name:
                new_node.attr[attr].CopyFrom(node.attr[attr])
                continue
            if node.attr[attr].type == types_pb2.DT_FLOAT:
                # modify node dtype
                node.attr[attr].type = dtype
            if attr == "value":
                tensor = node.attr[attr].tensor
                if tensor.dtype == types_pb2.DT_FLOAT:
                    # if float_val exists
                    if tensor.float_val:
                        float_val = tf.make_ndarray(node.attr[attr].tensor)
                        new_node.attr[attr].tensor.CopyFrom(tf.make_tensor_proto(float_val, dtype=dtype))
                        continue
                    # if tensor content exists
                    if tensor.tensor_content:
                        tensor_shape = [x.size for x in tensor.tensor_shape.dim]
                        tensor_weights = tf.make_ndarray(tensor)
                        # reshape tensor
                        tensor_weights = np.reshape(tensor_weights, tensor_shape)
                        tensor_proto = tf.make_tensor_proto(tensor_weights, dtype=dtype)
                        new_node.attr[attr].tensor.CopyFrom(tensor_proto)
                        continue
            new_node.attr[attr].CopyFrom(node.attr[attr])

    if not first_modify:
        return target_graph_def
    else:
        # transform graph
        if output_names:
            if not input_names:
                input_names = []
            transforms = ["strip_unused_nodes"]
            target_graph_def = TransformGraph(target_graph_def, input_names, output_names, transforms)
        # write graph_def to model
        tf.io.write_graph(target_graph_def, logdir=args.savepath, name=args.outpbname+'.pb', as_text=False)
        print("Converting done ...")

def check_fp16_model(model_path):
    print(model_path)
    """
    run this load_graph(model_path) to find error
    make sure the new fp16 graph can work 
    """
    new_sess = load_graph(model_path)

def fix_fp16_modelerror(model_path,targetop_name_expectedfp32=None,targetop_input_expectedfp32=None,targetop_name_expectedfp16=None,targetop_input_expectedfp16=None,input_names=[],output_names=[]):
    """
    If you find errors like the following, you can try the following solutions
    e.g.1
    error1:
    Input 1 of node tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize was passed half from 
    tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat:0 incompatible with expected float.
    error2:
    Input 1 of node tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/CropAndResize was passed half from 
    tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat:0 incompatible with expected float.

    opexpected_type is 'fp32', opinput_type is 'fp16'
    targetop_name_expectedfp32: ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize']
    targetop_input_expectedfp32: ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat']

    e.g.2
    error1:
    Input 0 of node tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transpose_1 was passed float from 
    tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize:0 incompatible with expected half.

    opexpected_type is 'fp16', opinput_type is 'fp32'
    targetop_name_expectedfp16: ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transpose_1']
    targetop_input_expectedfp16: ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize']
    """
    source_graph_def = convert_graph_all2fp16(model_path, target_type='fp16', first_modify=False)

    for node in source_graph_def.node:
        ## add cast node, opexpected_type='fp32', opinput_type='fp16'
        if node.name in targetop_input_expectedfp32: #'tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat'
            new_node_ = source_graph_def.node.add()
            new_node_.name = node.name+'_cast'
            new_node_.op = "Cast"
            new_node_.input.append(node.name)
            new_node_.attr["SrcT"].type = types_pb2.DT_HALF
            new_node_.attr["DstT"].type = types_pb2.DT_FLOAT
            new_node_.attr["Truncate"].b = False
            continue

        ## add cast node, opexpected_type='fp16', opinput_type='fp32'
        if node.name in targetop_input_expectedfp16: #'tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat'
            new_node_ = source_graph_def.node.add()
            new_node_.name = node.name+'_cast'
            new_node_.op = "Cast"
            new_node_.input.append(node.name)
            new_node_.attr["SrcT"].type = types_pb2.DT_FLOAT
            new_node_.attr["DstT"].type = types_pb2.DT_HALF
            new_node_.attr["Truncate"].b = False
            continue

    for node in source_graph_def.node:
        # convert targetop'input to cast node op name
        if (node.name in targetop_name_expectedfp32) or (node.name in targetop_name_expectedfp16):
            for i, name in enumerate(node.input):
                if (name in targetop_input_expectedfp32) or ( name in targetop_input_expectedfp16):
                    node.input[i]=name + '_cast'
            continue

    # transform graph
    if output_names:
        if not input_names:
            input_names = []
        transforms = ["strip_unused_nodes"]
        source_graph_def = TransformGraph(source_graph_def, input_names, output_names, transforms)
    # write graph_def to model
    tf.io.write_graph(source_graph_def, logdir=args.savepath, name=args.outpbname+'-new.pb', as_text=False)
    print("Converting new pb done ...")
##############################################################################################################################

## amp fp16-fp32 inference
##############################################################################################################################
def aotuamp_inference(modelpath,img_shape,input_names,output_names):
    ## in nvidia docker: docker pull nvcr.io/nvidia/tensorflow:20.12-tf1-py3
    ## docker run -itd -v /home/:/home/ --network host --gpus all --name ying.amp nvcr.io/nvidia/tensorflow:20.12-tf1-py3
    ## docker exec -it ying.amp /bin/bash
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"  
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1" 
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"] = "1" 
    os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LOG_PATH'] ="./amp_log/"

    graph_def = tf.GraphDef()
    if len(img_shape) == 3:
        img = np.random.rand(img_shape[0],img_shape[1],img_shape[2]).astype(np.float32)
    if len(img_shape) == 4:
        img = np.random.rand(img_shape[0],img_shape[1],img_shape[2],img_shape[3]).astype(np.float32)
    with open(modelpath, "rb") as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

        config = tf.ConfigProto()
        # method2
        # config.graph_options.rewrite_options.auto_mixed_precision=1
        with tf.Session(config=config) as sess:
            tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            # for i in tensor_name_list:
            #     print(i)

            input_tensorlist = []
            for input_tensor in input_names:
                input_tensorlist.append(sess.graph.get_tensor_by_name(input_tensor))
            output_tensorlist = []
            for output_tensor in output_names:
                output_tensorlist.append(sess.graph.get_tensor_by_name(output_tensor))
            print("output_tensorlist:",output_tensorlist)
            print("input_tensorlist[0]:",input_tensorlist[0])
            output = sess.run(output_tensorlist, feed_dict={input_tensorlist[0]: img})

            ## compute fps
            time_start =time.time()
            for i in range(1000):
                output = sess.run(output_tensorlist, feed_dict={input_tensorlist[0]: img})
            time_end =time.time()
            print('Inference fps:',1/((time_end-time_start)/1000))
##############################################################################################################################

if __name__ == "__main__":
    if args.countop:
        print("1、start countop:")
        count_op_numbers(args.modelpath)
    if args.ckpt2pb:
        print("2、start ckpt2pb:")
        ckpt2pb(ckptmodel_path='./ckptfiles/',ckptmodel_name='model-333',pbmodel_path='./pbfiles/',input_names=None,output_names=['tower-pred-0/output/boxes', 'tower-pred-0/output/scores','tower-pred-0/output/labels'])
    if args.checkef32:
        print("3、start checkef32:")
        check_ef32_model(args.modelpath)

    ## you can check inferf16 graph and inferaotuamp graph
    ## find the difference between manual fp16-graph generation and automatic fp16-graph generation
    if args.inferf16:
        print("4、start inferf16:")
        if not os.path.exists(args.savepath):
            os.mkdir(args.savepath)

        first_modify=False
        if first_modify==True:
            ## first step, you should run first_modify=True, turn all fp32 node to fp16 node
            ## you can get a basic pb structure of the FP16 datatype, e.g.fasterrcnn-fp16.pb
            convert_graph_all2fp16(args.modelpath, target_type=args.targettype, input_names=args.inputnames, output_names=args.outputnames)
            check_fp16_model(args.savepath+args.outpbname+'.pb')
        else:
            ## second step, you should check whether the e.g.fasterrcnn-fp16.pb can be run directly
            ## if not, find out what went wrong, git an example in func check_fp16_model()
            fix_fp16_modelerror(
                args.modelpath,\
                targetop_name_expectedfp32=['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/CropAndResize'],\
                targetop_input_expectedfp32=['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat'],\
                targetop_name_expectedfp16=['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transpose_1',\
                                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transpose_1',\
                                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/transpose_1',\
                                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/transpose_1'],\
                targetop_input_expectedfp16=['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/CropAndResize',\
                                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/CropAndResize'],\
                input_names=args.inputnames,\
                output_names=args.outputnames)
            check_fp16_model(args.savepath+args.outpbname+'-new.pb')
    if args.inferaotuamp:
        print("5、start inferaotuamp:")
        aotuamp_inference(args.modelpath,img_shape=[256,256,3],input_names=args.inputnames,output_names=args.outputnames)