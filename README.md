# Control

control: our controller package

- launch: launch files for controller function

- models: yolo models for object detection

- msg: custom ros messages used by controller functions

- scripts: controller functions in python

- src: controller functions in c++

- srv: custom ros services used by controller functions


# build control
add this in the cmakelist.txt to avoid NvInfer.h and cuda_runtime.h not found: https://github.com/mit-han-lab/inter-operator-scheduler/issues/3
include_directories(/home/{user}/TensorRT-8.6.1.6/include)
link_directories(/home/{user}/TensorRT-8.6.1.6/lib)

include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)"
