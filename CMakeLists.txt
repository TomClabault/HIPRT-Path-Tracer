cmake_minimum_required(VERSION 3.24)

project(HIPRTPathTracer LANGUAGES CXX)

# To be able to use the ExternalProject_Add() command
include(ExternalProject)
include(FetchContent)
# To see the progress of FetchContent
Set(FETCHCONTENT_QUIET FALSE)
# Policy for what timestamp to use when downloading stuff with FetchContent / ExternelProject / ...
# NEW sets the timestamps to the extraction time
cmake_policy(SET CMP0135 NEW)

# If the build type wasn't given on the commandline, we're defaulting to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Build type not specified: Using Debug by default")
endif()

# Sets up ASSIMP library CMake variable to prepare the building step
include (cmake/SetupASSIMP.cmake)

# Open Image Denoise binaries
include(cmake/SetupOIDN.cmake)

# Preparing HIPRT
include(cmake/SetupHIPRT.cmake)

# Preparing Orochi
include(cmake/SetupOrochi.cmake)


set(GLFW_LIB_DIR "thirdparties/opengl/lib/GLFW")
set(GLEW_LIB_DIR "thirdparties/opengl/lib/GLEW")
set(GLEW_BIN_DIR "thirdparties/opengl/bin/GLEW")

# Using CMake here to define C++ macros that will be used to find the directory of the kernels, etc...
# in the C++ code. This basically avoids hardcoding the path to the kernels in C++ and instead
# use the more flexible approach of defining it in the CMake
add_compile_definitions(DEVICE_KERNELS_DIRECTORY="../src/Device/kernels")
add_compile_definitions(DEVICE_INCLUDES_DIRECTORY="../src/") # This gives access to Device/ and HostDeviceCommon/
add_compile_definitions(OROCHI_INCLUDES_DIRECTORY="${OROCHI_SOURCES_DIR}/..") # This gives access to <Orochi/Orochi.h> in the kernels
add_compile_definitions(GLSL_SHADERS_DIRECTORY="../src/Shaders")

link_directories(${CMAKE_SOURCE_DIR}/${GLFW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLEW_LIB_DIR})

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE OPENGL_HEADERS thirdparties/opengl/include/*.h)
file(GLOB_RECURSE STBI_HEADERS thirdparties/stbi/*.h)
file(GLOB_RECURSE IMGUI_FILES thirdparties/imgui/*.h thirdparties/imgui/*.cpp)
file(GLOB_RECURSE GLM_FILES thirdparties/glm/*.h thirdparties/glm/*.cpp)
file(GLOB_RECURSE DEVICE_SOURCES src/Device/*.h)
file(GLOB_RECURSE GLSL_SHADERS src/Shaders/*.frag src/Shaders/*.vert)
file(GLOB_RECURSE HIPRT_HEADERS ${HIPRT_HEADERS_DIR}/*.h)
file(GLOB_RECURSE OROCHI_SOURCES_AND_HEADERS ${OROCHI_SOURCES_DIR}/*.h ${OROCHI_SOURCES_DIR}/*.cpp)
file(GLOB_RECURSE CUEW_SOURCES_AND_HEADERS ${CUEW_SOURCES_DIR}/*.h ${CUEW_SOURCES_DIR}/*.cpp)
file(GLOB_RECURSE HIPEW_SOURCES_AND_HEADERS ${HIPEW_SOURCES_DIR}/*.h ${HIPEW_SOURCES_DIR}/*.cpp)

add_executable(HIPRTPathTracer
	${SOURCE_FILES}

	${OPENGL_HEADERS}
	${STBI_HEADERS}
	${IMGUI_FILES}
	${ASSIMP_HEADERS}

	${DEVICE_SOURCES}
	${GLSL_SHADERS}
	${HIPRT_HEADERS}
	${OROCHI_SOURCES_AND_HEADERS}
	${CUEW_SOURCES_AND_HEADERS}
	${HIPEW_SOURCES_AND_HEADERS}
)

set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenImageDenoise REQUIRED HINTS ${oidnbinaries_SOURCE_DIR}) # HINTS to indicate a folder to search for the library in

if (WIN32)
	# "version" is a library from the Windows SDK
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32 ${hiprt_link_lib} version)
elseif(UNIX)
	find_package(GLEW REQUIRED)
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw GLEW::GLEW ${hiprt_link_lib})
endif()

target_include_directories(HIPRTPathTracer PRIVATE "src/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/opengl/include")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/glm/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/imgui/")
target_include_directories(HIPRTPathTracer PRIVATE ${HIPRT_HEADERS_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE ${OROCHI_SOURCES_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")

# ------------- Handling NVIDIA compilation details -------------
# If CUDA is installed, meaning NVIDIA
set(USING_NVIDIA False)

if (WIN32)
	if (DEFINED ENV{CUDA_PATH})
		set(USING_NVIDIA True)
	endif()
elseif (UNIX)
	# TODO find_package(CUDA) is deprecated and we may want to use enable_language(CUDA) + ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} if this becomes an issue
	# 	--> Setting the policy to OLD in the meantime
	cmake_policy(SET CMP0146 OLD)
	find_package(CUDA QUIET)
	if (CUDA_FOUND)
		set (USING_NVIDIA True)
	endif()
endif()

if (USING_NVIDIA)
	# Adding the required macro for Orochi
	add_compile_definitions(OROCHI_ENABLE_CUEW)

	# Adding the include directory for CUDA headers (needed for Orochi compilation)
	if (WIN32)
		target_include_directories(HIPRTPathTracer PRIVATE $ENV{CUDA_PATH}/include)
	elseif(UNIX)
		target_include_directories(HIPRTPathTracer PRIVATE ${CUDA_INCLUDE_DIRS})
	endif()
endif()
# ---------------------------------------------------------------


if (WIN32)
	message(STATUS "Copying OpenImageDenoise binaries...")
	file(GLOB OIDN_BINARIES ${oidnbinaries_SOURCE_DIR}/bin/*.dll)
	file(COPY ${OIDN_BINARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

	message(STATUS "Copying HIPRT binaries...")
	file(GLOB HIPRT_BINARIES ${HIPRT_BIN_DIR}/amd*.dll ${HIPRT_BIN_DIR}/hip*.dll)
	file(COPY ${HIPRT_BINARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

	message(STATUS "Copying Glew binaries...")
	file(COPY ${CMAKE_SOURCE_DIR}/${GLEW_BIN_DIR}/glew32.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()



if(MSVC_IDE)
	# Enabling parallel compilation on MSVC which isn't enabled by default
	if(MSVC)
		add_definitions(/MP)
	endif()


	# Macro to preserve nice beautiful source files hierarchy in Visual Studio
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
	
	# Creating a Visual Studio folder for the targets we don't care about so we have
	# a way to have our IDE look clean
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	set_target_properties(assimp uninstall zlibstatic UpdateAssimpLibsDebugSymbolsAndDLLs PROPERTIES FOLDER ExternalTargets)
endif()
