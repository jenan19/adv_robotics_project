cmake_minimum_required(VERSION 3.15)
project(visualHull)

set(CMAKE_CXX_STANDARD 17  )

##########################################
####                                  ####
####            FOR OPEN3D            ####
####               Start              ####
##########################################


#if(POLICY CMP0091)
    # https://stackoverflow.com/a/56490614
 #   cmake_policy(SET CMP0091 NEW)
#endif()

# The options need to be the same as Open3D's default
# If Open3D is configured and built with custom options, you'll also need to
# specify the same custom options.
#option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
# This needs cmake_policy(SET CMP0091 NEW)
#if (STATIC_WINDOWS_RUNTIME)
 #   set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
#else()
 #   set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
#endif()

#project(TestVisualizer LANGUAGES C CXX)

# Find installed Open3D, which exports Open3D::Open3D
#if(WIN32)
 #   find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
#else()
 #   find_package(Open3D HINTS ~/open3d_install/lib/cmake)
#endif()
#if(NOT Open3D_FOUND)
 #   message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
#endif()
	
##########################################
####                                  ####
####            FOR OPEN3D            ####
####               End                ####
##########################################



find_package(OpenCV REQUIRED)
#find_package(Open3D REQUIRED)

find_package(Eigen3 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${Open3D_INCLUDE_DIRS})
message(STATUS ${EIGEN3_INCLUDE_DIRS})
message(STATUS ${OpenCV_INCLUDE_DIRS})

message(STATUS ${Open3D_INCLUDE_DIRS})


add_executable(visualHull ../src/main.cpp)

target_link_libraries(visualHull  ${OpenCV_LIBS} )
