set(OROCHI_SUBMODULE_DIR ${CMAKE_SOURCE_DIR}/Orochi-Fork)

if(NOT EXISTS ${OROCHI_SUBMODULE_DIR}/ParallelPrimitives)
	# Making sure that the Orochi submodule was cloned
	message(FATAL_ERROR "The Orochi submodule couldn't be found. Did you forget to clone the submodules? Run 'git submodule update --init --recursive --jobs 10'.")
endif()

set(OROCHI_BIN_DIR ${OROCHI_SUBMODULE_DIR})
set(OROCHI_SOURCES_DIR ${OROCHI_SUBMODULE_DIR}/Orochi)

set(CUEW_SOURCES_DIR ${OROCHI_SUBMODULE_DIR}/contrib/cuew)
set(HIPEW_SOURCES_DIR ${OROCHI_SUBMODULE_DIR}/contrib/hipew)

STRING(REGEX REPLACE "\\\\" "/" OROCHI_SOURCES_DIR ${OROCHI_SOURCES_DIR})
STRING(REGEX REPLACE "\\\\" "/" CUEW_SOURCES_DIR ${CUEW_SOURCES_DIR})
STRING(REGEX REPLACE "\\\\" "/" HIPEW_SOURCES_DIR ${HIPEW_SOURCES_DIR})

# TODO remove when issue #7 (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/issues/7) is fixed
if (true)
	file(COPY ${OROCHI_SUBMODULE_DIR}/ParallelPrimitives DESTINATION ${CMAKE_SOURCE_DIR}/contrib/Orochi)
	file(COPY ${HIPRT_SUBMODULE_DIR}/hiprt/impl DESTINATION ${CMAKE_SOURCE_DIR}/hiprt)

	file(GLOB HIPRT_FILES_TO_COPY ${HIPRT_SUBMODULE_DIR}/hiprt/*.h ${HIPRT_SUBMODULE_DIR}/hiprt/*.in)
	file(COPY ${HIPRT_FILES_TO_COPY} DESTINATION ${CMAKE_SOURCE_DIR}/hiprt/)
endif()
