# ---------------- Welcom To Use DragonBenchmark  ----------------

PROJECT(reidentity)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)

# ---------------- Welcom To Use DragonBenchmark   ----------------


# ---------------- User Config ----------------
# Dependen Opencv 2.4.13.2
# 在这里修改替换为本机的opencv路径，版本要一致
set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/../3rdparty/opencv)

# -----For Developer
# 以下4行请在编译时注释掉
FILE(COPY ${PROJECT_SOURCE_DIR}/../include DESTINATION ${PROJECT_SOURCE_DIR})
message(STATUS "COPY " ${PROJECT_SOURCE_DIR}/../include " TO " ${PROJECT_SOURCE_DIR})
FILE(COPY ${PROJECT_SOURCE_DIR}/../lib DESTINATION ${PROJECT_SOURCE_DIR})
message(STATUS "COPY " ${PROJECT_SOURCE_DIR}/../lib " TO " ${PROJECT_SOURCE_DIR})


## 下面的不需要改动了
# ---[ Dependencies
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# ---[ Config types
set(CMAKE_BUILD_TYPE Release CACHE STRING "set build type to release")
set(CMAKE_CONFIGURATION_TYPES  Release CACHE STRING "set build type to release" FORCE)


# ---[ Includes
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OPENCV_DIR}/include)

# ---[ libs
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(${OPENCV_DIR}/lib)

# ---[ Install
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR} CACHE STRING "set install prefix" FORCE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${PROJECT_SOURCE_DIR}/lib ${OPENCV_DIR}/lib)

# ---[ defines

# ---[ Flags
if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -fpermissive -std=c++11")
endif()

# ---[ Warnings

# ---[ Utils
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/../bin)

# ---[ Files
FILE(GLOB_RECURSE SRC_FILES src/*.c src/*.cpp src/*.cu src/*.cc)

# ---[ complier
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_FILES})

# ---[ link basics
FILE(GLOB_RECURSE LIB_LIST ${PROJECT_SOURCE_DIR}/lib/*.so)
foreach(target ${LIB_LIST})
	TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${target})
endforeach()

TARGET_LINK_LIBRARIES(${PROJECT_NAME} opencv_core opencv_imgproc opencv_highgui opencv_video)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} stdc++fs pthread protobuf)

# ---[ install
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
install (TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_BINARY_DIR}/../bin)
