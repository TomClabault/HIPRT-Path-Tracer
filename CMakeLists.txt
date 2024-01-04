cmake_minimum_required(VERSION 3.20.5)

project(HIPRTPathTracer LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(OIDN_LIB_DIR "oidn/lib/")
set(THREADS_PREFER_PTHREAD_FLAG ON)

link_directories(${CMAKE_SOURCE_DIR}/${OIDN_LIB_DIR})

add_executable(HIPRTPathTracer

    source/bvh.cpp
    source/camera.cpp
    source/color.cpp
    source/flattened_bvh.cpp
    source/image_io.cpp
    source/main.cpp
    source/mat.cpp
    source/ray.cpp
    source/render_kernel.cpp
    source/tests.cpp
    source/triangle.cpp
    source/triangle.cpp
    source/utils.cpp
    source/vec.cpp

    include/bounding_volume.h
    include/bvh.h
    include/bvh_tests.h
    include/camera.h
    include/color.h
    include/flattened_bvh.h
    include/image.h
    include/image_io.h
    include/mat.h
    include/parsed_scene.h
    include/ray.h
    include/render_kernel.h
    include/sphere.h
    include/simple_material.h
    include/tests.h
    include/triangle.h
    include/utils.h
    include/vec.h
    include/xorshift.h

    stbi/stb_image.h
    stbi/stb_image_write.h
	
	tinygltf/tiny_gltf.h
    tinygltf/json.hpp

    rapidobj/rapidobj.hpp

    oidn/include/OpenImageDenoise/oidn.hpp
    oidn/include/OpenImageDenoise/oidn.h
    oidn/include/OpenImageDenoise/config.h)
	
set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)

find_package(OpenMP)
target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise)

target_include_directories(HIPRTPathTracer PRIVATE "include/")
target_include_directories(HIPRTPathTracer PRIVATE "stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "tinygltf/")
target_include_directories(HIPRTPathTracer PRIVATE "rapidobj/")
target_include_directories(HIPRTPathTracer PRIVATE "oidn/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")
