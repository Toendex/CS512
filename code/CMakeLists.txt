cmake_minimum_required(VERSION 2.8)
project( Panorama )
find_package( OpenCV REQUIRED )
add_executable( Panorama Panorama)
target_link_libraries( Panorama ${OpenCV_LIBS} )

