# from ultralytics import YOLO
import tensorrt as trt
# from ultralytics import YOLO
import onnx
import os
import lib.common as common
# model = YOLO('/media/slsecret/E624108524105B3F/Users/simon/Downloads/linxy/YOLOv8/model_city/best3_cococityyololg40/train7/weights/best.pt')  # initialize

# model.export(format='onnx')  # export

# Path to the ONNX model file
local_path = __file__.split("scripts")[0] + "/models/"
filename = "sissi11"
onnx_model_path = local_path + filename + ".onnx"

# Load the ONNX model
onnx_model = onnx.load(onnx_model_path)

# Load the PyTorch model
# model = torch.load('C:/Users/simon/Downloads/linxy/YOLOv8/try3_26/detect/train/weights/best.pt')
# Create a TensorRT engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
    common.EXPLICIT_BATCH
) as network, builder.create_builder_config() as config, trt.OnnxParser(
    network, TRT_LOGGER
) as parser, trt.Runtime(
    TRT_LOGGER
) as runtime:
    config.max_workspace_size = 1 << 28  # 256MiB
    builder.max_batch_size = 1
    engine = None

    if not parser.parse(onnx_model.SerializeToString()):
        print("ERROR: Failed to parse the ONNX file.")
    else:
        plan = builder.build_serialized_network(network,config)
        # engine = runtime.deserialize_cuda_engine(plan)
        with open(local_path+filename+'.engine', 'wb') as f:
            f.write(plan)