set(HIPRT_SUBMODULE_DIR ${CMAKE_SOURCE_DIR}/thirdparties/HIPRT-Fork)

if(NOT EXISTS ${HIPRT_SUBMODULE_DIR}/hiprt)
	# Making sure that the HIPRT submodule was cloned
	message(FATAL_ERROR "The HIPRT submodule couldn't be found. Did you forget to clone the submodules? Run 'git submodule update --init --recursive'.")
endif()

# Executing HIPRT's premake
if (WIN32)
	execute_process(COMMAND ./tools/premake5/win/premake5.exe vs2022 WORKING_DIRECTORY ${HIPRT_SUBMODULE_DIR})

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
	execute_process(COMMAND ${MSBUILD_EXECUTABLE} hiprt02003.vcxproj /property:Configuration=${CMAKE_BUILD_TYPE} /property:Platform=x64 /property:MultiProcessorCompilation=true /verbosity:minimal WORKING_DIRECTORY ${HIPRT_SUBMODULE_DIR}/build)
elseif(UNIX)
	execute_process(COMMAND ./tools/premake5/linux64/premake5 gmake WORKING_DIRECTORY ${HIPRT_SUBMODULE_DIR})

	set(HIPRT_MAKE_CONFIG_TYPE ${CMAKE_BUILD_TYPE})
	string(TOLOWER ${HIPRT_MAKE_CONFIG_TYPE} HIPRT_MAKE_CONFIG_TYPE)
	# config must take the value debug_x64, release_x64 or relwithdebinfo_x64. That's why we lower cased CMAKE_BUILD_TYPE just above
	execute_process(COMMAND make -C build -j config=${HIPRT_MAKE_CONFIG_TYPE}_x64 WORKING_DIRECTORY ${HIPRT_SUBMODULE_DIR})
endif()

if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
	set(hiprt_link_lib "hiprt0200364D")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
	set(hiprt_link_lib "hiprt0200364")
endif()

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
