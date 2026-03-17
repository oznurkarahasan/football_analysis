cmake_minimum_required(VERSION 3.16)
project(football_analysis_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn video)

add_library(football_analysis_core
    src/ball_trajectory_filler.cpp
    src/possession_assigner.cpp
    src/speed_distance_estimator.cpp
    src/pipeline.cpp
    src/camera_motion_estimator.cpp
)

target_include_directories(football_analysis_core PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(football_analysis_core PUBLIC ${OpenCV_LIBS})

add_executable(football_analysis_app
    src/main.cpp
    src/byte_track_lite.cpp
)
target_link_libraries(football_analysis_app PRIVATE football_analysis_core ${OpenCV_LIBS})
target_include_directories(football_analysis_app PRIVATE ${OpenCV_INCLUDE_DIRS})
