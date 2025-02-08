import argparse
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.tal import make_anchors
import torch
import numpy as np
import yaml

import onnx
import onnxsim
import onnx_graphsurgeon as gs

def parse_args():
    parser = argparse.ArgumentParser(description='Export yolo11 model trained on coco.')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--weights', dest='weights', type=str, default='./model/yolo11s.pt', help='Path of trained model weights.')
    parser.add_argument('--end2end', action="store_true", default=False, help='Attach tensorrt nms plugin node')
    parser.add_argument('--config', default='./config/plugin_config.yml', help='Plugin config file used in end2end')
    parser.add_argument('--dynamic_batch_size', action="store_true", default=False, help='Use dynamic batch size')
    parser.add_argument('--batch_size', type=int, default=1, help='Model input batch size')
    parser.add_argument('--input_h', type=int, default=720, help='Model\'s image input height')
    parser.add_argument('--input_w', type=int, default=1280, help='Model\'s image input width')
    parser.add_argument('--net_input_h', type=int, default=384, help='Model\'s network input height, make sure it\'s divided by 32')
    parser.add_argument('--net_input_w', type=int, default=640, help='Model\'s network input width, make sure it\'s divided by 32')
    parser.add_argument('--output', default='./model', help='Output onnx file path')
    return parser.parse_args()
    
class ModifiedDetectHead(Detect):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        # y = self._inference(x)
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        dbox = dbox.permute((0,2,1))
        cls_score = cls.sigmoid().permute(0,2,1)
        return dbox, cls_score

@gs.Graph.register()
def attach_nms_plugin(self, inputs, outputs, attr_config):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()

    # check at https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L413
    # and https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L443
    op_attrs = dict()
    op_attrs["score_threshold"] = attr_config["score_threshold"]
    op_attrs["iou_threshold"] = attr_config["iou_threshold"]
    op_attrs["max_output_boxes"] = attr_config["max_output_boxes"]
    op_attrs["background_class"] = attr_config["background_class"]
    op_attrs["score_activation"] = attr_config["score_activation"]
    op_attrs["class_agnostic"] = attr_config["class_agnostic"]
    op_attrs["box_coding"] = attr_config["box_coding"]

    return self.layer(name="EfficientNMSPlugin_0", op="EfficientNMS_TRT", inputs=inputs, outputs=outputs, attrs=op_attrs)

@gs.Graph.register()
def insert_cast_node(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()
    op_attrs = dict()
    op_attrs["to"] = 1
    return self.layer(name="Cast_0", op="Cast", inputs=inputs, outputs=outputs, attrs=op_attrs)

@gs.Graph.register()
def insert_precess_plugin(self, inputs, outputs, attr_config):
    for inp in inputs:
        inp.outputs.clear()
    for out in outputs:
        out.inputs.clear()

    # check at https://github.com/1punch3coins/yolov11_e2e/blob/main/src/nvprocess/img_precess_p.cpp#L167
    op_attrs = dict()
    op_attrs["src_crop"] = np.array(attr_config["src_crop"], dtype=np.uint32)    # the static input img crop location(left, top) and size(w, h)
    op_attrs["dst_crop"] = np.array(attr_config["dst_crop"], dtype=np.uint32)    # the static modle mat crop location(left, top) and size(w, h)
    op_attrs["src_size"] = np.array(attr_config["src_size"], dtype=np.uint32)    # the static input img size(h, w)
    op_attrs["dst_size"] = np.array(attr_config["dst_size"], dtype=np.uint32)    # the static model mat size(h, w)
    op_attrs["scale_inv"] = np.array(attr_config["scale_inv"], dtype=np.float32) # (x_axis, y_axis)
    op_attrs["mean"] = np.array(attr_config["mean"], dtype=np.float32)           # means(r, g, b)
    op_attrs["norm_inv"] = np.array(attr_config["norm_inv"], dtype=np.float32)   # norms(r, g, b)
    op_attrs["src_step"] = np.array(attr_config["src_step"], dtype=np.uint32)
    op_attrs["dst_step"] = np.array(attr_config["dst_step"], dtype=np.uint32)

    return self.layer(name="ImgPrecessPlugin_0", op="ImgPrecessPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

def attach_nms_node(onnx_model, batch_size, attr_config):
    print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
    graph = gs.import_onnx(onnx_model)
    N = batch_size
    L = attr_config["max_output_boxes"]

    # check at https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/README.md
    # and https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin/efficientNMSPlugin.cpp#L200
    out0 = gs.Variable(name="num", dtype=np.int32, shape=(N, 1))
    out1 = gs.Variable(name="boxes", dtype=np.float32, shape=(N, L, 4))
    out2 = gs.Variable(name="scores", dtype=np.float32, shape=(N, L))
    out3 = gs.Variable(name="classes", dtype=np.int32, shape=(N, L))

    graph.attach_nms_plugin(graph.outputs, [out0, out1, out2, out3], attr_config)
    graph.outputs = [out0, out1, out2, out3]
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def add_cast_node(onnx_model, batch_size, img_h, img_w):
    print("Use onnx_graphsurgeon to adjust preprocessing part in the onnx...")
    graph = gs.import_onnx(onnx_model)
    N = batch_size
    H = img_h
    W = img_w

    # 
    in0 = gs.Variable(name="input_img_uint8", dtype=np.uint8, shape=(N, H, W, 3))

    graph.insert_cast_node([in0], graph.inputs)
    graph.inputs = [in0]
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def add_precess_node(onnx_model, batch_size, img_h, img_w, attr_config):
    print("Use onnx_graphsurgeon to adjust preprocessing part in the onnx...")
    graph = gs.import_onnx(onnx_model)
    N = batch_size
    H = img_h
    W = img_w

    # 
    in0 = gs.Variable(name="input_img_float", dtype=np.float32, shape=(N, H, W, 3))

    graph.insert_precess_plugin([in0], graph.inputs, attr_config)
    graph.inputs = [in0]
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cpu" if args.cpu else "cuda")
    
    model = YOLO(args.weights).model.to(device)
    model.model[-1].export=True
    model.model[-1].format="onnx"
    if args.end2end:
        setattr(model.model[-1], '__class__', ModifiedDetectHead)
    model.eval()
    
    if args.dynamic_batch_size:
        onnx_file_path_prefix = args.output+"/yolov11s"+"_nx"+str(args.input_h)+"x"+str(args.input_w)+"x3("+str(args.net_input_h)+"x"+str(args.net_input_w)+")"
    else:
        onnx_file_path_prefix = args.output+"/yolov11s"+"_"+str(args.batch_size)+"x"+str(args.input_h)+"x"+str(args.input_w)+"x3("+str(args.net_input_h)+"x"+str(args.net_input_w)+")"
    onnx_file_path = onnx_file_path_prefix + "_raw.onnx"
    dummy_input = torch.randn((args.batch_size, 3, args.net_input_h, args.net_input_w)).to(device)
    if args.dynamic_batch_size:
        torch.onnx.export(
            model,                     # model to be exported
            dummy_input,               # example input tensor
            onnx_file_path,            # file where the model will be saved
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=15,          # ONNX version to export the model to
            do_constant_folding=True,  # whether to perform constant folding for optimization
            input_names=['input'],     # name of the input tensor
            output_names=['output'],   # name of the output tensor
            dynamic_axes={'input': {0: 'batch_size'}}
        )
    else:
        torch.onnx.export(
            model,                     # model to be exported
            dummy_input,               # example input tensor
            onnx_file_path,            # file where the model will be saved
            export_params=True,        # store the trained parameter weights inside the model file
            opset_version=15,          # ONNX version to export the model to
            do_constant_folding=True,  # whether to perform constant folding for optimization
            input_names=['input'],     # name of the input tensor
            output_names=['output'],   # name of the output tensor
        )
    print("nn part convertion completed")

    onnx_raw = onnx.load(onnx_file_path)
    onnx_simp, check = onnxsim.simplify(onnx_raw)
    print("onnx model simplification completed")
    with open(args.config, "r") as file:
        attr_config = yaml.load(file, Loader=yaml.SafeLoader)
    assert(attr_config['precess_plugin']['src_size'][0] == args.input_h)
    assert(attr_config['precess_plugin']['src_size'][1] == args.input_w)
    assert(attr_config['precess_plugin']['dst_size'][0] == args.net_input_h)
    assert(attr_config['precess_plugin']['dst_size'][1] == args.net_input_w)

    if args.end2end:
        if args.dynamic_batch_size:
            onnx_modified = add_precess_node(onnx_simp, 'batch_size', args.input_h, args.input_w, attr_config['precess_plugin'])
            onnx_modified = add_cast_node(onnx_modified, 'batch_size', args.input_h, args.input_w)
            onnx_final    = attach_nms_node(onnx_modified, 'batch_size', attr_config['nms_plugin'])
            onnx.save(onnx_final, onnx_file_path_prefix+".onnx")
        else:
            onnx_modified = add_precess_node(onnx_simp, args.batch_size, args.input_h, args.input_w, attr_config['precess_plugin'])
            onnx_modified = add_cast_node(onnx_modified, args.batch_size, args.input_h, args.input_w)
            onnx_final    = attach_nms_node(onnx_modified, args.batch_size, attr_config['nms_plugin'])
            onnx.save(onnx_final, onnx_file_path_prefix+".onnx")
    else:
        onnx.save(onnx_simp, onnx_file_path_prefix+".onnx")
    print("plugin nodes insertion completed")
    print("model saved to "+onnx_file_path_prefix+".onnx")