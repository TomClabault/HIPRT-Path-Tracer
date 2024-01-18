cmake_minimum_required(VERSION 3.20.5)

project(HIPRTPathTracer LANGUAGES CXX)

#TODO remove
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

#To be able to use the ExternalProject_Add() command
include(ExternalProject)

#Downloading ASSIMP library from Github for assets loading
set(EXTERNAL_ASSIMP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external/assimp)

ExternalProject_Add(assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_ASSIMP_INSTALL_LOCATION}
)

#TODO remove ?
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(OIDN_LIB_DIR "oidn/lib/")
set(OIDN_BIN_DIR "oidn/bin")

link_directories(${CMAKE_SOURCE_DIR}/${OIDN_LIB_DIR})
link_directories(${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib)

file(GLOB_RECURSE SOURCE_FILES source/*.cpp)
file(GLOB_RECURSE INCLUDE_FILES include/*.h)
file(GLOB_RECURSE STBI_HEADERS stbi/*.h)
file(GLOB_RECURSE OIDN_HEADERS oidn/include/OpenImageDenoise/*.hpp oidn/include/OpenImageDenoise/*.h)

add_executable(HIPRTPathTracer

	${SOURCE_FILES}
	${INCLUDE_FILES}

	${STBI_HEADERS}
	
	${ASSIMP_HEADERS}
	
	${OIDN_HEADERS}
)

add_dependencies(HIPRTPathTracer assimp)
	
set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)

find_package(OpenMP)
target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise)
if (WIN32)
    target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/assimp-vc143-mt.lib)
elseif(LINUX)
    target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/libassimp.so)
endif()

#TODO remove ?
target_include_directories(HIPRTPathTracer PRIVATE "include/")
target_include_directories(HIPRTPathTracer PRIVATE "stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "oidn/include/")
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")

if (CMAKE_GENERATOR MATCHES "Visual Studio")
	set_target_properties(HIPRTPathTracer 
        PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${EXTERNAL_ASSIMP_INSTALL_LOCATION}/bin;${OIDN_BIN_DIR};"
    )
endif()

