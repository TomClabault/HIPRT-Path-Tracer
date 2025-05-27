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

# Include Tracy for OpenGL profiling
include(cmake/SetupTracy.cmake)

set(GLFW_LIB_DIR "thirdparties/opengl/lib/GLFW")
set(GLEW_LIB_DIR "thirdparties/opengl/lib/GLEW")
set(GLEW_BIN_DIR "thirdparties/opengl/bin/GLEW")
set(TRACY_PUBLIC_DIR "thirdparties/tracy/public")

# Using CMake here to define C++ macros that will be used to find the directory of the kernels, etc...
# in the C++ code. This basically avoids hardcoding the path to the kernels in C++ and instead
# use the more flexible approach of defining it in the CMake
add_compile_definitions(DEVICE_KERNELS_DIRECTORY="../src/Device/kernels")
add_compile_definitions(DEVICE_INCLUDES_DIRECTORY="../src/") # This gives access to Device/ and HostDeviceCommon/
add_compile_definitions(OROCHI_INCLUDES_DIRECTORY="${OROCHI_SOURCES_DIR}/..") # This gives access to <Orochi/Orochi.h> in the kernels
add_compile_definitions(GLSL_SHADERS_DIRECTORY="../src/Shaders")
add_compile_definitions(DATA_DIRECTORY="${CMAKE_SOURCE_DIR}/data")
add_compile_definitions(BRDFS_DATA_DIRECTORY="${CMAKE_SOURCE_DIR}/data/BRDFsData")
#add_compile_definitions(TRACY_ENABLE="1")

link_directories(${CMAKE_SOURCE_DIR}/${GLFW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLEW_LIB_DIR})

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE OPENGL_HEADERS thirdparties/opengl/include/*.h)
file(GLOB_RECURSE STBI_HEADERS thirdparties/stbi/*.h)
# Selecting only what we need from the whole ImGui submodule
file(GLOB_RECURSE IMGUI_FILES thirdparties/imgui/imgui.h
	thirdparties/imgui/imgui.cpp
	thirdparties/imgui/imgui_demo.cpp
	thirdparties/imgui/imgui_draw.cpp
	thirdparties/imgui/imgui_tables.cpp
	thirdparties/imgui/imgui_widgets.cpp
	thirdparties/imgui/backends/imgui_impl_glfw.cpp
	thirdparties/imgui/backends/imgui_impl_opengl3.cpp
	thirdparties/imgui/misc/cpp/imgui_stdlib.cpp)

file(GLOB_RECURSE DEVICE_SOURCES src/Device/*.h)
file(GLOB_RECURSE GLSL_SHADERS src/Shaders/*.frag src/Shaders/*.vert)
file(GLOB_RECURSE HIPRT_HEADERS ${HIPRT_HEADERS_DIR}/*.h)
file(GLOB_RECURSE OROCHI_SOURCES_AND_HEADERS ${OROCHI_SOURCES_DIR}/*.h ${OROCHI_SOURCES_DIR}/*.cpp)
file(GLOB_RECURSE CUEW_SOURCES_AND_HEADERS ${CUEW_SOURCES_DIR}/*.h ${CUEW_SOURCES_DIR}/*.cpp)
file(GLOB_RECURSE HIPEW_SOURCES_AND_HEADERS ${HIPEW_SOURCES_DIR}/*.h ${HIPEW_SOURCES_DIR}/*.cpp)
file(GLOB_RECURSE TINYFILEDIALOGS_SOURCE_AND_HEADERS thirdparties/libtinyfiledialogs/tinyfiledialogs.h thirdparties/libtinyfiledialogs/tinyfiledialogs.c)

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
	${TINYFILEDIALOGS_SOURCE_AND_HEADERS}
)

set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenImageDenoise REQUIRED HINTS ${oidnbinaries_SOURCE_DIR}) # HINTS to indicate a folder to search for the library in

if (WIN32)
	# "version" is a library from the Windows SDK
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32 hiprt02004 TracyClient version)
elseif(UNIX)
	find_package(GLEW REQUIRED)
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw GLEW::GLEW hiprt02004 TracyClient)
endif()

target_include_directories(HIPRTPathTracer PRIVATE "src/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/opengl/include")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/libtinyfiledialogs/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/glm/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/imgui/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/imgui/backends")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/tinyexr/")
target_include_directories(HIPRTPathTracer PRIVATE ${HIPRT_HEADERS_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE ${OROCHI_SOURCES_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")
target_include_directories(HIPRTPathTracer PRIVATE ${TRACY_PUBLIC_DIR})

# Auto setup of Orochi for NVIDIA by including their cmake file
include(${HIPRT_SUBMODULE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)


if (WIN32)
	message(STATUS "Copying OpenImageDenoise binaries...")
	file(GLOB OIDN_BINARIES ${oidnbinaries_SOURCE_DIR}/bin/*.dll)
	file(COPY ${OIDN_BINARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

	# For copying HIPRT's DLL
	get_target_property(HIPRT_DLL_NAME hiprt02004 OUTPUT_NAME)

	if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND MSVC_IDE)
		# Adding the 'd' suffix that MSVC adds to debug libraries file names
		set(HIPRT_DLL_NAME ${HIPRT_DLL_NAME}d)
	endif()
	# Appending .dll extension
	set(HIPRT_DLL_NAME ${HIPRT_DLL_NAME}.dll)

	add_custom_command(OUTPUT
        ${CMAKE_BINARY_DIR}/${HIPRT_DLL_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hiprt02004>
        ${CMAKE_BINARY_DIR}/${HIPRT_DLL_NAME}
    DEPENDS hiprt02004)
	# Create target which consume the command via DEPENDS.
	add_custom_target(hiprtCopyDLL ALL DEPENDS ${CMAKE_BINARY_DIR}/${HIPRT_DLL_NAME})
	add_dependencies(HIPRTPathTracer hiprtCopyDLL)

	message(STATUS "Copying Glew binaries...")
	file(COPY ${CMAKE_SOURCE_DIR}/${GLEW_BIN_DIR}/glew32.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()



if(MSVC_IDE)
	# Enabling parallel compilation on MSVC which isn't enabled by default
	if(MSVC)
		add_definitions(/MP)

		if (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
			# In RelWithDebInfo, we're disabling all optimizations
			# for easier debugging:
			#	- RelWithDebInfo is faster than debug so we want to use
			#	that when debugging line by line
			#	
			#	- But the optimizations that are enabled by default in
			#	RelWithDebInfo still mess up with MSVC debugger and the
			#	debugger jumps everywhere, variables are optimized away etc...
			#	Even if we're supposed to include debug infos in the compilation...
			#	So we're just disabling optimizations then

			# No optimizations
			add_definitions(/Od)

			# No inlining of functions
			add_definitions(/Ob0)
		endif()
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
	set_target_properties(
		assimp uninstall zlibstatic UpdateAssimpLibsDebugSymbolsAndDLLs # ASSIMP Targets
		hiprt02004 hiprtCopyDLL # HIPRT Targets
		TracyClient # Tracy
		PROPERTIES FOLDER ExternalTargets)
endif()
