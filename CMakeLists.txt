# cmake_minimum_required(VERSION 3.0.2)
cmake_minimum_required(VERSION 3.18)
project(control)

set(CMAKE_CXX_STANDARD 17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  message_generation
  sensor_msgs
  cv_bridge
  image_transport
  utils
  cuda
  # roslib
  # image_geometry
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED )
# Set ncnn_DIR using the relative path
set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/cmake/ncnn)
# find_package(TensorRT REQUIRED)
# if (TensorRT_FOUND)
#     message("TensorRT found ${TensorRT_VERSION}, ${TensorRT_INCLUDE_DIRS}")
# else()
#     message("TensorRT not found")
# endif()


# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR /home/cyrus/work/libs/TensorRT-8.6.1.6/)
endif()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})

add_executable(run_inference_benchmark src/main.cpp)
target_link_libraries(run_inference_benchmark tensorrt_cpp_api)




find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("OpenCV found ${OpenCV_VERSION}, ${OpenCV_INCLUDE_DIRS}")
else()
    message("OpenCV not found")
endif()

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Lane.msg
#   Sign.msg
# )

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Localization.srv
  nav.srv
  get_direction.srv
  dotted.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs#   tf2_geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES control
  CATKIN_DEPENDS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  utils
  message_runtime
  image_transport
  cv_bridge
  sensor_msgs
  opencv2
  #  roslib
  #  image_geometry
  #  message_generation
  #  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  "include"
  src/include 
  include/ncnn
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/control/src/include
  ${OpenCV_INCLUDE_DIRS}
  # ${CMAKE_CURRENT_SOURCE_DIR}/src/control/src/include
)

# include_directories(/home/slsecret/TensorRT-8.6.1.6/include)
# link_directories(/home/slsecret/TensorRT-8.6.1.6/lib)

# include_directories(/usr/local/cuda/targets/x86_64-linux/include)
# link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

# include_directories(/usr/local/include/opencv2)
# link_directories(/usr/local/lib)

# add_executable(camera
#   src/camera.cpp
# )
# add_executable(cameraExample
#   src/cameraExample.cpp
#   src/yolo-fastestv2.cpp
# )
# add_executable(lane
#   src/lane.cpp
# )
# add_executable(sign
#   src/sign.cpp
# )
# add_executable(signFastest
#   src/signFastest.cpp
#   src/yolo-fastestv2.cpp
# )
add_executable(signTRT2
src/signTRT2.cpp
src/engine.cpp
src/yolo-fastestv2.cpp
)
add_executable(bridge
  src/localization_bridge.cpp
)
# add_executable(cameraNode
#   src/cameraNode.cpp
#   src/include/raspicam/raspicam_cv.cpp
#   src/yolo-fastestv2.cpp
# )
# add_executable(cameraNode2
#   src/cameraNode2.cpp
#   # src/include/raspicam/raspicam_cv.cpp
#   src/yolo-fastestv2.cpp
# )
# add_executable(localization
#   src/localization.cpp
# )

# target_link_libraries(camera
#   ${catkin_LIBRARIES}
# )
# target_link_libraries(cameraExample
#   ${catkin_LIBRARIES}
#   ncnn ${OpenCV_LIBS}
# )
# target_link_libraries(lane
#   ${catkin_LIBRARIES}
# )
# target_link_libraries(sign
#   ${catkin_LIBRARIES}
# )
# target_link_libraries(signFastest
#   ${catkin_LIBRARIES}
#   ncnn ${OpenCV_LIBS}
# )
target_link_libraries(bridge
  ${catkin_LIBRARIES}
)
target_link_libraries(signTRT2
  ${catkin_LIBRARIES}
)
# target_link_libraries(cameraNode
#   ${catkin_LIBRARIES}
#   ncnn ${OpenCV_LIBS}
#   raspicam
# )
# target_link_libraries(cameraNode2
#   ${catkin_LIBRARIES}
#   ncnn ${OpenCV_LIBS}
#   # raspicam
# )
# target_link_libraries(localization
#   ${catkin_LIBRARIES}
# )
# find_package(OpenMP)
# if (OpenMP_CXX_FOUND)
#     target_link_libraries(signFastest PUBLIC OpenMP::OpenMP_CXX)
# endif()

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  scripts/lane.py
  scripts/sign_onnx.py
  scripts/sign_tensorrt.py
  # scripts/realSensePub.py
  # scripts/trackObjectDistance.py
  scripts/control3.py
  # scripts/localization.py
  scripts/encoderSim.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

