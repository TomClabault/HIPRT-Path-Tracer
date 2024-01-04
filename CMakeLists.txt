if (CMAKE_HOST_WIN32)
    # need at least CMake 3.23 for IntelLLVM support of IntelDPCPP package on Windows
    cmake_minimum_required(VERSION 3.23)
else()
    # CMake 3.20.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.20.5)
endif()

project(SYCL_RayTracing LANGUAGES CXX)

#find_package(IntelDPCPP REQUIRED)

file(GLOB SYCL_HEADERS "/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl/*.hpp" "/opt/intel/oneapi/compiler/2023.2.1/linux/include/sycl/CL/*.hpp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

set(OIDN_LIB_DIR "oidn/lib/")
set(THREADS_PREFER_PTHREAD_FLAG ON)

link_directories(${CMAKE_SOURCE_DIR}/${OIDN_LIB_DIR})

add_executable(SYCL_RayTracing
    ${SYCL_HEADERS}

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
    include/parsed_obj.h
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
	
set_property(TARGET SYCL_RayTracing PROPERTY CXX_STANDARD 20)

find_package(OpenMP)
target_link_libraries(SYCL_RayTracing PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise)

target_include_directories(SYCL_RayTracing PRIVATE "/opt/intel/oneapi/compiler/2023.2.1/linux/include/")
target_include_directories(SYCL_RayTracing PRIVATE "include/")
target_include_directories(SYCL_RayTracing PRIVATE "stbi/")
target_include_directories(SYCL_RayTracing PRIVATE "rapidobj/")
target_include_directories(SYCL_RayTracing PRIVATE "oidn/include/")
target_include_directories(SYCL_RayTracing PRIVATE ".")
