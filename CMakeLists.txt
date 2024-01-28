cmake_minimum_required(VERSION 3.20.5)

project(HIPRTPathTracer LANGUAGES CXX)

#To be able to use the ExternalProject_Add() command
include(ExternalProject)

#Downloading ASSIMP library from Github for assets loading
set(EXTERNAL_ASSIMP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external/assimp)

# If the build type wasn't given on the commandline, we're defaulting to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Build type not specified: Using Debug by default")
endif()

# TODO always build ASSIMP in release to avoid the issue of the debug vs release .lib file with Visual Studio on Windows?
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

set(HIPRT_LIB_DIR "thirdparties/hiprt/lib")
set(HIPRT_BIN_DIR "thirdparties/hiprt/bin")

set(KERNELS_DIR "src/Kernels")

link_directories(${CMAKE_SOURCE_DIR}/${OIDN_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLFW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLEW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${HIPRT_LIB_DIR})
link_directories(${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE OPENGL_HEADERS thirdparties/opengl/include/*.h)
file(GLOB_RECURSE STBI_HEADERS thirdparties/stbi/*.h)
file(GLOB_RECURSE OIDN_HEADERS thirdparties/oidn/include/OpenImageDenoise/*.hpp thirdparties/oidn/include/OpenImageDenoise/*.h)
file(GLOB_RECURSE HIPRT_HEADERS thirdparties/hiprt/include/*.h)
file(GLOB_RECURSE OROCHI_FILES thirdparties/Orochi/*.h thirdparties/Orochi/*.cpp)

add_executable(HIPRTPathTracer

	${SOURCE_FILES}
	${INCLUDE_FILES}

	${OPENGL_HEADERS}
	${STBI_HEADERS}
	${ASSIMP_HEADERS}
	${OIDN_HEADERS}
	${HIPRT_HEADERS}
	${OROCHI_FILES}
)

set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)
add_dependencies(HIPRTPathTracer assimp)

find_package(OpenMP)
find_package(OpenGL REQUIRED)

target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32 hiprt0200264)

# "version" is a library from the Windows SDK
target_link_libraries(HIPRTPathTracer PRIVATE version)

if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	# Linking with the debug library of ASSIMP
	target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/assimp-vc143-mtd.lib)
	
	# We're also adding the command to copy ASSIMP's DLL
	add_custom_command(TARGET HIPRTPathTracer POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/bin/assimp-vc143-mtd.dll ${CMAKE_CURRENT_BINARY_DIR}
	)
else()
	# Linking with the release library of ASSIMP
	target_link_libraries(HIPRTPathTracer PRIVATE ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib/assimp-vc143-mt.lib)
	
	# We're also adding the command to copy ASSIMP's DLL
	add_custom_command(TARGET HIPRTPathTracer POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${EXTERNAL_ASSIMP_INSTALL_LOCATION}/bin/assimp-vc143-mt.dll ${CMAKE_CURRENT_BINARY_DIR}
	)
endif()

target_include_directories(HIPRTPathTracer PRIVATE "src/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/opengl/include")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/oidn/include/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/hiprt/include/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/Orochi/")
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")

add_custom_command(TARGET HIPRTPathTracer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${OIDN_BIN_DIR}/OpenImageDenoise.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${OIDN_BIN_DIR}/OpenImageDenoise_core.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${OIDN_BIN_DIR}/OpenImageDenoise_device_cpu.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${OIDN_BIN_DIR}/OpenImageDenoise_device_cuda.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${OIDN_BIN_DIR}/OpenImageDenoise_device_hip.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${GLEW_BIN_DIR}/glew32.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${HIPRT_BIN_DIR}/hiprt0200264.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${HIPRT_BIN_DIR}/hiprtc0507.dll ${CMAKE_CURRENT_BINARY_DIR}
	COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/Kernels
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/${KERNELS_DIR} ${CMAKE_CURRENT_BINARY_DIR}/Kernels
)

# Macro to preserve source files hierarchy in Visual Studio
if(MSVC_IDE)
	macro(GroupSources curdir)
		file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)

		foreach(child ${children})
			if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
				GroupSources(${curdir}/${child})
			else()
				string(REPLACE "/" "\\" groupname ${curdir})
				string(REPLACE "src" "Sources" groupname ${groupname})
				source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			endif()
		endforeach()
	endmacro()

	# Run macro
	GroupSources(src)
endif()

