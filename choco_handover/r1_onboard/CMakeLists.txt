cmake_minimum_required(VERSION 3.10)
project(choco_handover_arm_daemon LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(/home/unitree/unitree_sdk2 sdk2)

add_executable(r1_arm_daemon r1_arm_daemon.cpp)
target_link_libraries(r1_arm_daemon unitree_sdk2 pthread)
