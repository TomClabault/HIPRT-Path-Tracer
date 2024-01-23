cmake_minimum_required(VERSION 3.20.5)

project(HIPRTPathTracer LANGUAGES CXX)

#To be able to use the ExternalProject_Add() command
include(ExternalProject)

#Downloading ASSIMP library from Github for assets loading
set(EXTERNAL_ASSIMP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external/assimp)

ExternalProject_Add(assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_ASSIMP_INSTALL_LOCATION}
)

set(OIDN_LIB_DIR "thirdparties/oidn/lib/")
set(OIDN_BIN_DIR "thirdparties/oidn/bin")

set(GLFW_LIB_DIR "thirdparties/opengl/lib/GLFW")
set(GLFW_BIN_DIR "thirdparties/opengl/bin/GLFW")

set(GLEW_LIB_DIR "thirdparties/opengl/lib/GLEW")
set(GLEW_BIN_DIR "thirdparties/opengl/bin/GLEW")

link_directories(${CMAKE_SOURCE_DIR}/${OIDN_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLFW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLEW_LIB_DIR})
link_directories(${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib)

file(GLOB_RECURSE SOURCE_FILES source/*.cpp source/*.h)
file(GLOB_RECURSE OPENGL_HEADERS thirdparties/opengl/include/*.h)
file(GLOB_RECURSE STBI_HEADERS thirdparties/stbi/*.h)
file(GLOB_RECURSE OIDN_HEADERS thirdparties/oidn/include/OpenImageDenoise/*.hpp thirdparties/oidn/include/OpenImageDenoise/*.h)

add_executable(HIPRTPathTracer

	${SOURCE_FILES}
	${INCLUDE_FILES}

	${OPENGL_HEADERS}
	${STBI_HEADERS}
	${ASSIMP_HEADERS}
	${OIDN_HEADERS}
)

set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)
add_dependencies(HIPRTPathTracer assimp)

find_package(OpenMP)
find_package(OpenGL REQUIRED)
target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32)
if (WIN32)
    target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/assimp-vc143-mt.lib)
elseif(LINUX)
    target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/libassimp.so)
endif()

target_include_directories(HIPRTPathTracer PRIVATE "source/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/opengl/include")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/oidn/include/")
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")

if (CMAKE_GENERATOR MATCHES "Visual Studio")
	#Adding the DLL folder of ASSIMP to visual studio so that the project can run from Visual Studio editor without screaming "Missing DLL...."
	set_target_properties(HIPRTPathTracer 
        PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${EXTERNAL_ASSIMP_INSTALL_LOCATION}/bin;${OIDN_BIN_DIR};${GLEW_BIN_DIR};"
    )
endif()

