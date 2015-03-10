cmake_minimum_required(VERSION 2.8)
project( OpticalFlow )
find_package( OpenCV REQUIRED )
add_executable( OpticalFlow osx.cpp Timer.cpp)
target_link_libraries( OpticalFlow ${OpenCV_LIBS} )

