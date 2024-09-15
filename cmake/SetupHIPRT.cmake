set(HIPRT_SUBMODULE_DIR ${CMAKE_SOURCE_DIR}/thirdparties/HIPRT-Fork)

if(NOT EXISTS ${HIPRT_SUBMODULE_DIR}/hiprt)
	# Making sure that the HIPRT submodule was cloned
	message(FATAL_ERROR "The HIPRT submodule couldn't be found. Did you forget to clone the submodules? Run 'git submodule update --init --recursive'.")
endif()

set(NO_ENCRYPT ON)
set(NO_UNITTEST ON)
#option(HIPRT_PREFER_HIP_5 "Prefer HIP 5" OFF)

set(CMAKE_EXE_LINKER_FLAGS_DEBUGGPU "")
add_subdirectory(${HIPRT_SUBMODULE_DIR})

# Now that we built HIPRT, we can set the variables that will be used in the rest of the CMake 
# to find the headers, the libraries, ...
set(HIPRT_BIN_DIR ${HIPRT_SUBMODULE_DIR}/dist/bin/${CMAKE_BUILD_TYPE})
set(HIPRT_HEADERS_DIR ${HIPRT_SUBMODULE_DIR}/hiprt)

# The GPU compiler will need this additional include folder to properly compile some kernels
add_compile_definitions(KERNEL_COMPILER_ADDITIONAL_INCLUDE="${HIPRT_SUBMODULE_DIR}")

# Replacing backslashes in the Windows paths that lead to wrong escape character
# note that the four backslashes \\\\ are required because we need a regular expression that
# compiles to '\'.
# \\ is converted by CMake to a single '\'
# so \\\\ is converted by CMake to '\\' which is the regular expression for the single '\' character
STRING(REGEX REPLACE "\\\\" "/" HIPRT_HEADERS_DIR ${HIPRT_HEADERS_DIR})
	
link_directories(${HIPRT_BIN_DIR})
