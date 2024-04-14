cmake_minimum_required(VERSION 3.20.5)

project(HIPRTPathTracer LANGUAGES CXX)

# To be able to use the ExternalProject_Add() command
include(ExternalProject)
include(FetchContent)

# Downloading ASSIMP library from Github for assets loading
set(EXTERNAL_ASSIMP_INSTALL_LOCATION ${CMAKE_BINARY_DIR}/external/assimp)

# If the build type wasn't given on the commandline, we're defaulting to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
  message(STATUS "Build type not specified: Using Debug by default")
endif()

# TODO always build ASSIMP in release to avoid the issue of the debug vs release .lib file with Visual Studio on Windows? : use -DCMAKE_BUILD_TYPE=Release
ExternalProject_Add(
	assimp
    GIT_REPOSITORY https://github.com/assimp/assimp
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_ASSIMP_INSTALL_LOCATION}
)

# ----------- Downloading Open Image Denoise binaries -----------
FetchContent_Declare(
	oidnbinaries
	URL      https://github.com/OpenImageDenoise/oidn/releases/download/v2.2.2/oidn-2.2.2.x64.windows.zip
	FIND_PACKAGE_ARGS NAMES OpenImageDenoise
)

FetchContent_MakeAvailable(
	oidnbinaries
)
# ---------------------------------------------------------------

# ----------------- Requiring the HIPRT Binaries ----------------
if (NOT DEFINED HIPRT_BINARIES_DIR)
	set(HIPRT_BINARIES_DIR "NOT FOUND" CACHE PATH "Path to the HIPRT binaries directory. Your should provide the path such that HIPRT_BINARIES_DIR/hiprt/win/hiprt0200264.lib is a valid path.")
	message(FATAL_ERROR "The path to the binaries folder of HIPRT is needed. Provide it with the HIPRT_BINARIES_DIR variable using 'cmake -DHIPRT_BINARIES_DIR=<path>'. HIPRT binaries can be downloaded at https://gpuopen.com/hiprt/")
elseif (DEFINED HIPRT_BINARIES_DIR AND NOT EXISTS ${HIPRT_BINARIES_DIR}/hiprt/win/hiprt0200364D.lib)
	message(FATAL_ERROR "The HIPRT binaries directory you provided does not contain hiprt0200264.lib under hiprt/win. Your should provide HIPRT_BINARIES_DIR such that HIPRT_BINARIES_DIR/hiprt/win/hiprt0200264.lib is a valid path.")
else ()
	set(HIPRT_BIN_DIR ${HIPRT_BINARIES_DIR}/hiprt/win)
	set(HIPRT_HEADERS_DIR ${HIPRT_BINARIES_DIR}/hiprt)

	set(OROCHI_BIN_DIR ${HIPRT_BINARIES_DIR}/contrib/Orochi/)
	set(OROCHI_SOURCES_DIR ${HIPRT_BINARIES_DIR}/contrib/Orochi/Orochi)

	set(CUEW_SOURCES_DIR ${HIPRT_BINARIES_DIR}/contrib/Orochi/contrib/cuew)
	set(HIPEW_SOURCES_DIR ${HIPRT_BINARIES_DIR}/contrib/Orochi/contrib/hipew)

	# Replacing backslashes in the Windows paths that lead to wrong escape character
	# note that the four backslashes \\\\ are required because we need a regular expression that
	# compiles to '\'.
	# \\ is converted by CMake to a single '\'
	# so \\\\ is converted by CMake to '\\' which is the regular expression for the single '\' character
	STRING(REGEX REPLACE "\\\\" "/" HIPRT_HEADERS_DIR ${HIPRT_HEADERS_DIR})
	STRING(REGEX REPLACE "\\\\" "/" OROCHI_SOURCES_DIR ${OROCHI_SOURCES_DIR})
	STRING(REGEX REPLACE "\\\\" "/" CUEW_SOURCES_DIR ${CUEW_SOURCES_DIR})
	STRING(REGEX REPLACE "\\\\" "/" HIPEW_SOURCES_DIR ${HIPEW_SOURCES_DIR})

	link_directories(${HIPRT_BINARIES_DIR}/hiprt/win)
endif()
# ---------------------------------------------------------------

set(GLFW_LIB_DIR "thirdparties/opengl/lib/GLFW")
set(GLFW_BIN_DIR "thirdparties/opengl/bin/GLFW")

set(GLEW_LIB_DIR "thirdparties/opengl/lib/GLEW")
set(GLEW_BIN_DIR "thirdparties/opengl/bin/GLEW")

set(KERNELS_DIR "src/Kernels")
set(SHADERS_DIR "src/Shaders")
set(HOST_DEVICE_COMMON_DIR "src/HostDeviceCommon")

link_directories(${CMAKE_SOURCE_DIR}/${GLFW_LIB_DIR})
link_directories(${CMAKE_SOURCE_DIR}/${GLEW_LIB_DIR})
link_directories(${EXTERNAL_ASSIMP_INSTALL_LOCATION}/lib)
link_directories(CMAKE_CURRENT_BINARY_DIR)

file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.h)
file(GLOB_RECURSE OPENGL_HEADERS thirdparties/opengl/include/*.h)
file(GLOB_RECURSE STBI_HEADERS thirdparties/stbi/*.h)
file(GLOB_RECURSE IMGUI_FILES thirdparties/imgui/*.h thirdparties/imgui/*.cpp)
file(GLOB_RECURSE GLM_FILES thirdparties/glm/*.h thirdparties/glm/*.cpp)
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

	${HIPRT_HEADERS}
	${OROCHI_SOURCES_AND_HEADERS}
	${CUEW_SOURCES_AND_HEADERS}
	${HIPEW_SOURCES_AND_HEADERS}
)

set_property(TARGET HIPRTPathTracer PROPERTY CXX_STANDARD 20)
add_dependencies(HIPRTPathTracer assimp)

find_package(OpenMP)
find_package(OpenGL REQUIRED)
find_package(OpenImageDenoise REQUIRED HINTS ${oidnbinaries_SOURCE_DIR})

# "version" is a library from the Windows SDK
target_link_libraries(HIPRTPathTracer PRIVATE OpenMP::OpenMP_CXX OpenImageDenoise ${OPENGL_LIBRARY} glfw3 glew32 hiprt0200364D version)

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


# TODO instead of copying, try to have the app use these as env variable ? Basically have the DLL on the path ?
message(STATUS "Copying OpenImageDenoise binaries...")
configure_file(${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise.dll ${CMAKE_CURRENT_BINARY_DIR}/OpenImageDenoise.dll COPYONLY)
configure_file(${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise_core.dll ${CMAKE_CURRENT_BINARY_DIR}/OpenImageDenoise_core.dll COPYONLY)
configure_file(${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise_device_cpu.dll ${CMAKE_CURRENT_BINARY_DIR}/OpenImageDenoise_device_cpu.dll COPYONLY)
configure_file(${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise_device_cuda.dll ${CMAKE_CURRENT_BINARY_DIR}/OpenImageDenoise_device_cuda.dll COPYONLY)
configure_file(${oidnbinaries_SOURCE_DIR}/bin/OpenImageDenoise_device_hip.dll ${CMAKE_CURRENT_BINARY_DIR}/OpenImageDenoise_device_hip.dll COPYONLY)

add_custom_command(TARGET HIPRTPathTracer PRE_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/${GLEW_BIN_DIR}/glew32.dll ${CMAKE_CURRENT_BINARY_DIR}
)

# The function below is using configure_file just to make sure that the file is copied to the binary dir everytime it has changed. 
# This is important for shaders as they are not part of the CPP compilation process. Thus, if we were only using an
# approach such as add_custom_command to copy the shader files, the shaders would not be copied to the binary directory when they
# modified since the project wouldn't be rebuilt (because no file that require re-compilation (i.e. cpp files) were modified).
# configure_file ensures the input it is given is copied (or configured but we're only using this command for its
# copy capability here) to the output everytime it is modified, independant of whether a compilation was triggered or not.
function (copy_folder input_folder output_folder extensions)
	file(GLOB input_files_to_copy ${input_folder}/${extensions})
	
	foreach(file_to_copy ${input_files_to_copy})
		cmake_path(GET file_to_copy FILENAME filename_to_copy)
		configure_file(${file_to_copy} ${output_folder}/${filename_to_copy} COPYONLY)
	endforeach()
endfunction()

# Copying the kernel files to runtime directory so that they are available for runtime compilation (by Orochi)
copy_folder("${CMAKE_SOURCE_DIR}/${KERNELS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/Kernels" "*.*")
copy_folder("${CMAKE_SOURCE_DIR}/${KERNELS_DIR}/includes" "${CMAKE_CURRENT_BINARY_DIR}/Kernels/includes" "*.*")
copy_folder("${CMAKE_SOURCE_DIR}/${SHADERS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/Shaders" "*.*")
copy_folder("${CMAKE_SOURCE_DIR}/${HOST_DEVICE_COMMON_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/HostDeviceCommon" "*.*")

# Copying hiprt binaries
message(STATUS "Copying HIPRT binaries...")
copy_folder("${HIPRT_BIN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" "*.dll")
#copy_folder("${HIPRT_BIN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" "*.hipfb")
copy_folder("${HIPRT_BIN_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" "*.bc")

# Copying Orochi kernels bitcodes
#copy_folder("${OROCHI_BIN_DIR}/bitcodes" "${CMAKE_CURRENT_BINARY_DIR}" "*.*")
#copy_folder("${OROCHI_BIN_DIR}/contrib/bin/win64" "${CMAKE_CURRENT_BINARY_DIR}" "*.dll")

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
endif()

