cmake_minimum_required(VERSION 3.0.2)
project(control)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

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
  turtlesim
  message_generation
  sensor_msgs
  cv_bridge
  image_transport
  # roslib
  # image_geometry
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Lane.msg
  Sign.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  Localization.srv
  nav.srv
  get_direction.srv
  dotted.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs#   tf2_geometry_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

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
 turtlesim
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

###########
## Build ##
###########

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

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /home/simonli/Documents/Simulator/src/control/src/lib/cmake/ncnn)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /home/simonli/Documents/Simulator/src/control/src/lib/cmake/ncnn)


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

find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
)

## Declare a C++ library

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(lane ${PROJECT_NAME}_generate_messages_cpp)
# add_dependencies(sign ${PROJECT_NAME}_generate_messages_cpp)
# add_dependencies(localization ${PROJECT_NAME}_gencpp)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(control2
#   src/control2.cpp
# )
# add_executable(statemachine
#   src/statemachine.cpp
# )
add_executable(camera
  src/camera.cpp
)
add_executable(cameraExample
  src/cameraExample.cpp
  src/yolo-fastestv2.cpp
)
add_executable(lane
  src/lane.cpp
)
add_executable(sign
  src/sign.cpp
)
add_executable(signFastest
  src/signFastest.cpp
  src/yolo-fastestv2.cpp
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

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies( signFastest ros_ncnn_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(control2
#   ${catkin_LIBRARIES}
# )
# target_link_libraries(statemachine
#   ${catkin_LIBRARIES}
# )
target_link_libraries(camera
  ${catkin_LIBRARIES}
)
target_link_libraries(cameraExample
  ${catkin_LIBRARIES}
  ncnn ${OpenCV_LIBS}
)
target_link_libraries(lane
  ${catkin_LIBRARIES}
)
target_link_libraries(sign
  ${catkin_LIBRARIES}
)
target_link_libraries(signFastest
  ${catkin_LIBRARIES}
  ncnn ${OpenCV_LIBS}
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
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
catkin_install_python(PROGRAMS
  scripts/lane.py
  scripts/sign_onnx.py
  # scripts/realSensePub.py
  # scripts/trackObjectDistance.py
  scripts/control3.py
  # scripts/localization.py
  scripts/encoderSim.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_control.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
