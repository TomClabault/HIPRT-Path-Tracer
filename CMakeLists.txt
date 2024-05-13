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


# ----------- Downloading ASSIMP library for assets loading -----------
FetchContent_Declare(
	assimplib
	GIT_REPOSITORY https://github.com/assimp/assimp
	GIT_TAG        8b9ed34eaa3e6ad24254cb7e058fb9150f66b865 # Release 5.4.0
	GIT_PROGRESS TRUE
)

# We're going to disable shared libs for assimp but we need to save
# the current value of BUILD_SHARED_LIBS before overriding it with
# OFF (for assimp only)
set(BUILD_SHARED_LIBS_BACKUP ${BUILD_SHARED_LIBS})
set(CMAKE_BUILD_TYPE_BACKUP ${CMAKE_BUILD_TYPE})
set(BUILD_SHARED_LIBS OFF)
set(ASSIMP_NO_EXPORT ON)
set(ASSIMP_BUILD_TESTS OFF)
set(ASSIMP_INSTALL_PDB OFF)
set(ASSIMP_BUILD_ZLIB ON)
set(ASSIMP_BUILD_ASSIMP_VIEW OFF)
FetchContent_MakeAvailable(
	assimplib
)

# Restoring varaibles
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACKUP})
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BACKUP})
# ---------------------------------------------------------------------




# ----------- Downloading Open Image Denoise binaries -----------
if (WIN32)
	set(OIDN_URL https://github.com/OpenImageDenoise/oidn/releases/download/v2.2.2/oidn-2.2.2.x64.windows.zip)
elseif(UNIX)
	set(OIDN_URL https://github.com/OpenImageDenoise/oidn/releases/download/v2.2.2/oidn-2.2.2.x86_64.linux.tar.gz)
endif()

FetchContent_Declare(
	oidnbinaries
	URL      ${OIDN_URL}
)

FetchContent_MakeAvailable(
	oidnbinaries
)
# ---------------------------------------------------------------




# ----------------- Downloading and building HIPRT ----------------
FetchContent_Declare(
  hiprtrelease
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git
  GIT_TAG        9aae35514303b6e3e99a4bf632e9be0fbfd208ad # release v2.3.7df94af + 2 commits that fix some stuff
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(
	hiprtrelease
)

# Executing HIPRT's premake
if (WIN32)
	execute_process(COMMAND ./tools/premake5/win/premake5.exe vs2022 WORKING_DIRECTORY ${hiprtrelease_SOURCE_DIR})

	# We're now going to build the HIPRT project through MSBuild since Premake generated a Visual Studio Solution
	# First, trying to find MSBuild on the PATH 
	find_program(MSBUILD_EXECUTABLE "MSBuild")
	if(NOT ${MSBUILD_EXECUTABLE} STREQUAL MSBUILD_EXECUTABLE-NOTFOUND AND MSBUILD_EXECUTABLE MATCHES "MSBuild.exe$")
		message("MSBuild found at: " ${MSBUILD_EXECUTABLE})
	else()
		# Trying to automatically find MSBuild in the regular Visual Studio installation
		# given by the VS2022INSTALLDIR environment variable
		if (EXISTS $ENV{VS2022INSTALLDIR}/MSBuild/Current/Bin/MSBuild.exe)
			set(MSBUILD_EXECUTABLE $ENV{VS2022INSTALLDIR}/MSBuild/Current/Bin/MSBuild.exe)
		else()
			set(MSBUILD_EXECUTABLE "NOT FOUND" CACHE PATH "Path to MSBuild.exe. MSBuild is shipped with Visual Studio and can be find under <Visual Studio Installation>/MSBuild/Current/Bin")
			message(FATAL_ERROR "Could not find MSBuild.exe on your system. It is shipped with Visual Studio so you should have it somewhere on your system in Visual Studio installation directory. It is usually found under <Visual Studio Installation>/MSBuild/Current/Bin. You can add it to your PATH or provide it through the MSBUILD_EXECUTABLE CMake variable.")
		endif()
	endif()

	# Building HIPRT through MSBuild since Premake generated a Visual Studio Solution
	execute_process(COMMAND ${MSBUILD_EXECUTABLE} hiprt02003.vcxproj /property:Configuration=Debug /property:Platform=x64 /property:MultiProcessorCompilation=true /verbosity:minimal WORKING_DIRECTORY ${hiprtrelease_SOURCE_DIR}/build)
elseif(UNIX)
	execute_process(COMMAND ./tools/premake5/linux64/premake5 gmake WORKING_DIRECTORY ${hiprtrelease_SOURCE_DIR})
	execute_process(COMMAND make -C build -j config=debug_x64 WORKING_DIRECTORY ${hiprtrelease_SOURCE_DIR})
endif()

# Now that we built HIPRT, we can set the variables that will be used in the rest of the CMake 
# to find the headers, the libraries, ...
set(HIPRT_BIN_DIR ${hiprtrelease_SOURCE_DIR}/dist/bin/Debug)
set(HIPRT_HEADERS_DIR ${hiprtrelease_SOURCE_DIR}/hiprt)

# The GPU compiler will need this additional include folder to properly compile some kernels
add_compile_definitions(KERNEL_COMPILER_ADDITIONAL_INCLUDE="${hiprtrelease_SOURCE_DIR}")

# Replacing backslashes in the Windows paths that lead to wrong escape character
# note that the four backslashes \\\\ are required because we need a regular expression that
# compiles to '\'.
# \\ is converted by CMake to a single '\'
# so \\\\ is converted by CMake to '\\' which is the regular expression for the single '\' character
STRING(REGEX REPLACE "\\\\" "/" HIPRT_HEADERS_DIR ${HIPRT_HEADERS_DIR})
	
link_directories(${HIPRT_BINARIES_DIR}/hiprt/win)
# -------------------------------------------------




# ----------------- Downloading Orochi (no need to build) ----------------

FetchContent_Declare(
	orochilib
	GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/Orochi.git
	GIT_TAG        17c5b9885549c1266f34af702f52ce9bdfdeae21 # Release 2.0.0
	GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(
	orochilib
)

set(OROCHI_BIN_DIR ${orochilib_SOURCE_DIR})
set(OROCHI_SOURCES_DIR ${orochilib_SOURCE_DIR}/Orochi)

set(CUEW_SOURCES_DIR ${orochilib_SOURCE_DIR}/contrib/cuew)
set(HIPEW_SOURCES_DIR ${orochilib_SOURCE_DIR}/contrib/hipew)

STRING(REGEX REPLACE "\\\\" "/" OROCHI_SOURCES_DIR ${OROCHI_SOURCES_DIR})
STRING(REGEX REPLACE "\\\\" "/" CUEW_SOURCES_DIR ${CUEW_SOURCES_DIR})
STRING(REGEX REPLACE "\\\\" "/" HIPEW_SOURCES_DIR ${HIPEW_SOURCES_DIR})

# TODO remove when issue #7 (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/issues/7) is fixed
if (true)
	file(COPY ${orochilib_SOURCE_DIR}/ParallelPrimitives DESTINATION ${CMAKE_SOURCE_DIR}/contrib/Orochi)
	file(COPY ${hiprtrelease_SOURCE_DIR}/hiprt/impl DESTINATION ${CMAKE_SOURCE_DIR}/hiprt)

	file(GLOB HIPRT_FILES_TO_COPY ${hiprtrelease_SOURCE_DIR}/hiprt/*.h ${hiprtrelease_SOURCE_DIR}/hiprt/*.in)
	file(COPY ${HIPRT_FILES_TO_COPY} DESTINATION ${CMAKE_SOURCE_DIR}/hiprt/)
endif()

# ------------------------------------------------------------------------





set(GLFW_LIB_DIR "thirdparties/opengl/lib/GLFW")
set(GLFW_BIN_DIR "thirdparties/opengl/bin/GLFW")

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
link_directories(${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib)
link_directories(${HIPRT_BIN_DIR})

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

find_package(OpenMP)
find_package(OpenGL REQUIRED)
find_package(OpenImageDenoise REQUIRED HINTS ${oidnbinaries_SOURCE_DIR}) # HINTS to indicate a folder to search for the library in

# "version" is a library from the Windows SDK
if (WIN32)
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32 hiprt0200364D version)
elseif(UNIX)
	find_package(GLEW REQUIRED)
	target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX assimp OpenImageDenoise ${OPENGL_LIBRARY} glfw GLEW::GLEW hiprt0200364D)
endif()

target_include_directories(HIPRTPathTracer PRIVATE "src/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/opengl/include")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/stbi/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/glm/")
target_include_directories(HIPRTPathTracer PRIVATE "thirdparties/imgui/")
# TODO using /.. here isn't super cool because it gives access to folders in the HIPRT SDK such as tutorials, tools, ...
# However, because we want to include the HIPRT and Orochi headers as <hiprt/*.h> and <Orochi/*.h>, we need to give
# the include directory as being one folder above the actual hiprt/ and Orochi/ folders that contain the includes
# Without the .., we would have to include the headers as <hiprt.h> in our source files instead of <hiprt/hiprt.h>
# which is less explicit and not wanted
# This is however the simplest approach to allow prefixing the headers with the folder name
target_include_directories(HIPRTPathTracer PRIVATE ${HIPRT_HEADERS_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE ${OROCHI_SOURCES_DIR}/..)
target_include_directories(HIPRTPathTracer PRIVATE "${EXTERNAL_ASSIMP_INSTALL_LOCATION}/include/")
target_include_directories(HIPRTPathTracer PRIVATE ".")

# ------------- Handling NVIDIA compilation details -------------
# If CUDA is installed, meaning NVIDIA
if(DEFINED ENV{CUDA_PATH})
	# Adding the required macro for Orochi
	add_compile_definitions(OROCHI_ENABLE_CUEW)
	target_include_directories(HIPRTPathTracer PRIVATE $ENV{CUDA_PATH}/include)
endif()
# ---------------------------------------------------------------


if (WIN32)
	message(STATUS "Copying OpenImageDenoise binaries...")
	file(GLOB OIDN_BINARIES ${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise*.dll)
	file(COPY ${OIDN_BINARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

	message(STATUS "Copying HIPRT binaries...")
	file(GLOB HIPRT_BINARIES ${HIPRT_BIN_DIR}/amd*.dll ${HIPRT_BIN_DIR}/hip*.dll)
	file(COPY ${HIPRT_BINARIES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

	message(STATUS "Copying Glew binaries...")
	file(COPY ${CMAKE_SOURCE_DIR}/${GLEW_BIN_DIR}/glew32.dll DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()



# Macro to preserve nice beautiful source files hierarchy in Visual Studio
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
	
	# Creating a Visual Studio folder for the targets we don't care about so we have
	# a way to have our IDE look clean
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	set_target_properties(assimp uninstall zlibstatic UpdateAssimpLibsDebugSymbolsAndDLLs PROPERTIES FOLDER ExternalTargets)
endif()
