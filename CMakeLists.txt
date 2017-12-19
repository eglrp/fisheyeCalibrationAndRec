cmake_minimum_required(VERSION 2.8.3)
project(fisheyeCalibration)

#SET( CMAKE_CXX_COMPILER "g++") #设定编译器

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages



# 增加opencv的依赖
FIND_PACKAGE( OpenCV 3.3.0 REQUIRED )










INCLUDE_DIRECTORIES(  ${OpenCV_INCLUDE_DIRS}  )




ADD_EXECUTABLE(fisheyeCalibration src/fisheye_calibration.cpp)
TARGET_LINK_LIBRARIES( fisheyeCalibration
${OpenCV_LIBS} 

)

