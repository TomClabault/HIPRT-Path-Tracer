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

set(ASSIMP_SUBMODULE_DIR ${CMAKE_SOURCE_DIR}/ASSIMP-Fork)

if(NOT EXISTS ${ASSIMP_SUBMODULE_DIR}/code)
	# Making sure that the HIPRT submodule was cloned
	message(FATAL_ERROR "The ASSIMP submodule couldn't be found. Did you forget to clone the submodules? Run 'git submodule update --init --recursive --jobs 10'.")
endif()

add_subdirectory(${ASSIMP_SUBMODULE_DIR})

# Restoring varaibles
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACKUP})
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE_BACKUP})