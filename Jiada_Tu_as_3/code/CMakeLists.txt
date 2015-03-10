cmake_minimum_required(VERSION 2.8)
project( findPoint )
find_package( OpenCV REQUIRED )
add_executable( findPoint findPoint.cpp)
target_link_libraries( findPoint ${OpenCV_LIBS} )

project( calibration )
find_package( OpenCV REQUIRED )
add_executable( calibration calibration.cpp)
target_link_libraries( calibration ${OpenCV_LIBS} )

project( calibration_opencv )
find_package( OpenCV REQUIRED )
add_executable( calibration_opencv calibration_opencv.cpp)
target_link_libraries( calibration_opencv ${OpenCV_LIBS} )

