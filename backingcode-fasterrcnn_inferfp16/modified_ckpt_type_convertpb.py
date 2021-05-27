import tensorflow as tf
from tensorflow.core.framework import types_pb2, graph_pb2, attr_value_pb2
from tensorflow.tools.graph_transforms import TransformGraph
from google.protobuf import text_format
import numpy as np
import sys
import os

'''
clean version 2020.10.13
'''

# Const should be float32 in object detection api during nms (see here: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/non-max-suppression-v4.html)
keep_fp32_node_name = ["conv0/bn/gamma/read","conv0/bn/beta/read","conv0/bn/mean/EMA/read","conv0/bn/variance/EMA/read",
                        "conv0/bn/gamma","conv0/bn/beta","conv0/bn/mean/EMA","conv0/bn/variance/EMA",]

# softmax node and  the following nodes
keep_fp32_softmax_nodes_name = ['tower-pred-0/fastrcnn_all_scores',
                            'tower-pred-0/output/strided_slice_1','tower-pred-0/output/transpose_1','tower-pred-0/output/Greater',
                            'tower-pred-0/output/Where','tower-pred-0/output/Slice','tower-pred-0/output/Max','tower-pred-0/output/add',
                            'tower-pred-0/output/Cast','tower-pred-0/output/mul','tower-pred-0/output/add_1','tower-pred-0/output/GatherNd_1',
                            'tower-pred-0/output/strided_slice_2','tower-pred-0/output/non_max_suppression/NonMaxSuppressionV3',
                            'tower-pred-0/output/scores','tower-pred-0/output/GatherV2','tower-pred-0/output/labels','tower-pred-0/output/strided_slice',
                            'tower-pred-0/output/GatherNd','tower-pred-0/output/boxes','tower-pred-0/output/Greater/y','tower-pred-0/output/add/y']

# while mul node appear overflow
keep_fp32_mul_nodes_name = ['tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/mul',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/Squeeze',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/Sqrt',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/mul',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/add',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/Log',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/mul_1',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/add_1',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/Floor',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/Cast',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/mul/y',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/add/y',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/mul_1/y',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/mul_1',
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/add_1/x']

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

# deal with FusedBatchNorm
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
    dtype = types_pb2.DT_FLOAT

    new_node = graph_def.node.add()
    new_node.op = "FusedBatchNormV2"
    new_node.name = node.name

    # keep node *"bn/FusedBatchNorm", input float32
    if "bn/FusedBatchNorm" in node.name:
        for i, name in enumerate(node.input):
            # print("i-",i,"-name-",name)
            if i==0:
                new_node.input.append(name+'_cast')
            else:
                new_node.input.append(name)
    else:
        new_node.input.extend(node.input)

    new_node.attr["U"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
    for attr in list(node.attr.keys()):
        if attr == "T":
            node.attr[attr].type = dtype
        new_node.attr[attr].CopyFrom(node.attr[attr])
    # print("rewrite fused_batch_norm done!")
    
    # result float32 to float16
    if "bn/FusedBatchNorm" in node.name:
        new_node_2 = graph_def.node.add()
        node_name = node.name+'_cast'
        new_node_2.name = node_name
        new_node_2.op = "Cast"
        new_node_2.input.append(node.name)
        new_node_2.attr["SrcT"].type = types_pb2.DT_FLOAT
        new_node_2.attr["DstT"].type = types_pb2.DT_HALF 
        new_node_2.attr["Truncate"].b = False

def convert_graph_to_fp16(model_path, save_path, outpb_name, as_text=False, target_type='fp16', input_name=None, output_names=None):
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

        # for node input
        if 'bn/output' in  node.name:
            new_node.input.append(node.name[:-6] + "FusedBatchNorm_cast")

        elif node.name in ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize',\
                           'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/CropAndResize',\
                           'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/CropAndResize',
                           'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/CropAndResize']:
            # step1 add fp32 '**_cast'
            for i, name in enumerate(node.input):
                if name in ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat'  ]:
                    new_node.input.append(name + '_cast')
                else:
                    new_node.input.append(name)
            
            for attr in list(node.attr.keys()):
                if attr == "T":
                    node.attr[attr].type = dtype
                new_node.attr[attr].CopyFrom(node.attr[attr])
            # step2 cast output to fp16
            new_node_2 = target_graph_def.node.add()
            node_name = node.name+'_cast'
            new_node_2.name = node_name
            new_node_2.op = "Cast"
            new_node_2.input.append(node.name)
            new_node_2.attr["SrcT"].type = types_pb2.DT_FLOAT 
            new_node_2.attr["DstT"].type = types_pb2.DT_HALF 
            new_node_2.attr["Truncate"].b = False
            continue

        elif node.name in ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transpose_1',\
                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transpose_1',\
                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/transpose_1',\
                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/transpose_1',\
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/mul',\
                            'tower-pred-0/fastrcnn_all_scores',\
                            'tower-pred-0/output/strided_slice'] :
            for i, name in enumerate(node.input):
                if name in ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/CropAndResize',\
                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/CropAndResize',\
                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/CropAndResize',\
                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/CropAndResize',\
                                'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/sub',\
                                    'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/sub_1',\
                                        'tower-pred-0/fastrcnn/outputs/class/output',\
                                            'tower-pred-0/output/transpose']:
                    new_node.input.append(name+'_cast')
                else:
                    new_node.input.append(name)
        else:
            new_node.input.extend(node.input)

        attrs = list(node.attr.keys())
        # keep batch norm params node
        if ("BatchNorm" in node.name) or ('batch_normalization' in node.name):
            for attr in attrs:
                new_node.attr[attr].CopyFrom(node.attr[attr])
            continue

        # replace dtype in node attr with target dtype
        for attr in attrs:
            # keep special node in fp32
            if node.name in keep_fp32_node_name or \
                node.name in keep_fp32_softmax_nodes_name or\
                node.name in keep_fp32_mul_nodes_name or \
                'bn/gamma' in node.name or "bn/beta" in node.name or "bn/mean" in node.name or "bn/variance" in node.name or \
                'non_max_suppression/iou_threshold' in node.name or 'non_max_suppression/score_threshold' in node.name:
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

        # add cast node
        if 'Conv2D' in node.name and 'tower-pred-0' in node.name:
            new_node_ = target_graph_def.node.add()
            node_name = node.name+'_cast'
            new_node_.name = node_name
            new_node_.op = "Cast"
            new_node_.input.append(node.name)
            new_node_.attr["SrcT"].type = types_pb2.DT_HALF
            new_node_.attr["DstT"].type = types_pb2.DT_FLOAT
            new_node_.attr["Truncate"].b = False

        # add cast node 
        if node.name in ['tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level3/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level4/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/roi_level5/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat',\
                            'tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/sub_1','tower-pred-0/multilevel_roi_align/fpn_map_rois_to_levels/area/sub',\
                            'tower-pred-0/fastrcnn/outputs/class/output',\
                            'tower-pred-0/output/transpose']:
            new_node_ = target_graph_def.node.add()
            new_node_.name = node.name+'_cast'
            new_node_.op = "Cast"
            new_node_.input.append(node.name)
            new_node_.attr["SrcT"].type = types_pb2.DT_HALF
            new_node_.attr["DstT"].type = types_pb2.DT_FLOAT
            new_node_.attr["Truncate"].b = False
          
    # transform graph
    if output_names:
        if not input_name:
            input_name = []
        transforms = ["strip_unused_nodes"]
        target_graph_def = TransformGraph(target_graph_def, input_name, output_names, transforms)
    # write graph_def to model
    tf.io.write_graph(target_graph_def, logdir=save_path, name=outpb_name, as_text=as_text)
    print("Converting done ...")

if __name__ == "__main__":
    # params
    # find pb in D:\git_ZhangXiaoying0116\fp16_inference\pbtemp
    model_path = "pbtemp/model-314435.pb" # source pb
    save_path = "pbtemp/" # target path
    outpb_name = "ckpt_type_convertpb_fasterrcnn_finalversion.pb" # target pb
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    as_text = False
    target_type = 'fp16'

    # object detection api input and output nodes
    input_name=['image:0']
    output_names=['tower-pred-0/output/boxes', 'tower-pred-0/output/scores','tower-pred-0/output/labels','tower-pred-0/conv0/Conv2D_cast','tower-pred-0/conv0/bn/FusedBatchNorm_cast','tower-pred-0/multilevel_roi_align/roi_level2/roi_align/crop_and_resize/transform_fpcoor_for_tf/concat_cast']
    convert_graph_to_fp16(model_path, save_path, outpb_name, as_text=as_text, target_type=target_type, input_name=input_name, output_names=output_names)
    
    # test loading
    # ISSUE: loading detection model is extremely slow while loading classification model is normal
    sess = load_graph(save_path+"/"+outpb_name)