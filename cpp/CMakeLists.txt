cmake_minimum_required(VERSION 2.8)
project(main)

#add_compile_options(-std=c++11)

#set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libtorch12/share/cmake/Torch)
#set(OpenCV_DIR /home/XXX/software/OpenCV320/share/OpenCV)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_executable(main main.cpp)
target_link_libraries(main
	${OpenCV_LIBS}
	${TORCH_LIBRARIES}
)
