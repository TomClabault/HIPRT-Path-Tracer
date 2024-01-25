#pragma once

#define HIPRT_MAJOR_VERSION 2
#define HIPRT_MINOR_VERSION 2
#define HIPRT_PATCH_VERSION 0x0e68f54

#define HIPRT_API_VERSION 2002
#define HIPRT_VERSION_STR "02002"
#define HIP_VERSION_STR "5.7"

#include <hiprt/hiprt_types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined( _MSC_VER )
#ifdef HIPRT_EXPORTS
#define HIPRT_API __declspec( dllexport )
#else
#define HIPRT_API __declspec( dllimport )
#endif
#elif defined( __GNUC__ )
#ifdef HIPRT_EXPORTS
#define HIPRT_API __attribute__( ( visibility( "default" ) ) )
#else
#define HIPRT_API
#endif
#else
#define HIPRT_API
#pragma warning Unknown dynamic link import / export semantics.
#endif

/** \brief Creates HIPRT API context.
 *
 * All HIPRT functions expect context as their first argument. Context
 * keeps global data required by HIPRT session. Calls made from different
 * threads with different HIPRT contexts are safe. Calls with the same context
 * should be externally synchronized by the client.
 *
 * \param hiprtApiVersion The API version.
 * \param contextOut The created context.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtCreateContext( uint32_t hiprtApiVersion, const hiprtContextCreationInput& input, hiprtContext& contextOut );

/** \brief Destorys the HIPRT API context.
 *
 * Destroys all the global resources used by HIPRT session. Further calls
 * with this context are prohibited.
 *
 * \param context The API context.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyContext( hiprtContext context );

/** \brief Creates a geometry.
 *
 * This function creates
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param geometryOut The resulting geometry.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateGeometry(
	hiprtContext				   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtGeometry&				   geometryOut );

/** \brief Creates geometries.
 *
 * This function creates an array of
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param numGeometries The number of geometries to be created.
 * \param buildInputs An array of build input structs.
 * \param buildOptions Various flags controlling build process.
 * \param geometriesOut The resulting geometries.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateGeometries(
	hiprtContext				   context,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputs,
	const hiprtBuildOptions		   buildOptions,
	hiprtGeometry**				   geometriesOut );

/** \brief Destroys a geometry.
 *
 * This function destroys
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param geometry The geometry.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyGeometry( hiprtContext context, hiprtGeometry geometry );

/** \brief Destroys geometries.
 *
 * This function destroys an array of
 * hiprtGeometry representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param numGeometries The number of geometries to be created.
 * \param geometry The geometries.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyGeometries( hiprtContext context, uint32_t numGeometries, hiprtGeometry* geometries );

/** \brief Builds or updates a geometry.
 *
 * Given geometry description from the client, this function builds
 * hiprtGeometry representing acceleration structure topology (in case of a
 * build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context The HIPRT API context.
 * \param buildOperation The type of the build operation.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param temporaryBuffer A temporary buffer for the build.
 * \param stream A stream used for the build.
 * \param geometryOut Resulting geometry.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildGeometry(
	hiprtContext				   context,
	hiprtBuildOperation			   buildOperation,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	hiprtApiStream				   stream,
	hiprtGeometry				   geometryOut );

/** \brief Builds or updates geometries.
 *
 * Given geometry description from the client, this function builds
 * hiprtGeometry representing acceleration structure topology (in case of a
 * build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context The HIPRT API context.
 * \param buildOperation The type of the build operation.
 * \param numGeometries The number of geometries to be built.
 * \param buildInputs An array of build input structs.
 * \param buildOptions Various flags controlling build process.
 * \param temporaryBuffer A temporary buffer for the build.
 * \param stream A stream used for the build.
 * \param geometriesOut Resulting geometries.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildGeometries(
	hiprtContext				   context,
	hiprtBuildOperation			   buildOperation,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputs,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	hiprtApiStream				   stream,
	hiprtGeometry*				   geometriesOut );

/** \brief Returns temporary storage requirements for geometry build.
 *
 * \param context The HIPRT API context.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param sizeOut The resulting size.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetGeometryBuildTemporaryBufferSize(
	hiprtContext context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions, size_t& sizeOut );

/** \brief Returns temporary storage requirements for geometry build.
 *
 * \param context The HIPRT API context.
 * \param numGeometries The number of geometries to be built.
 * \param buildInputs An array of build input structs.
 * \param buildOptions Various flags controlling build process.
 * \param sizeOut The resulting size.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetGeometriesBuildTemporaryBufferSize(
	hiprtContext				   context,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputs,
	const hiprtBuildOptions		   buildOptions,
	size_t&						   sizeOut );

/** \brief Compacts geometry.
 *
 * This function compacts hiprtGeometry, reducing used memory.
 * The input geometry are automatically destroyed.
 *
 * \param context The HIPRT API context.
 * \param stream A stream used for the compaction.
 * \param geometryIn The input geometry to be compacted.
 * \param geometryOut The compacted geometry.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtCompactGeometry( hiprtContext context, hiprtApiStream stream, hiprtGeometry geometryIn, hiprtGeometry& geometryOut );

/** \brief Compacts geometries.
 *
 * This function compacts an array of
 * hiprtGeometry, reducing used memory.
 * The input geometries are automatically destroyed.
 *
 * \param context The HIPRT API context.
 * \param numGeometries The number of geometries to be compacted.
 * \param stream A stream used for the compaction.
 * \param geometriesIn The input geometries to be compacted.
 * \param geometriesOut The compacted geometries.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCompactGeometries(
	hiprtContext	context,
	uint32_t		numGeometries,
	hiprtApiStream	stream,
	hiprtGeometry*	geometriesIn,
	hiprtGeometry** geometriesOut );

/** \brief Creates a scene.
 *
 * This function creates
 * hiprtScene representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param sceneOut The resulting scene.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateScene(
	hiprtContext context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions, hiprtScene& sceneOut );

/** \brief Creates scenes.
 *
 * This function creates
 * hiprtScene representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param numScenes The number of scenes to be created.
 * \param buildInputs An array of build input structs.
 * \param buildOptions Various flags controlling build process.
 * \param scenesOut The resulting scenes.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateScenes(
	hiprtContext				context,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInputs,
	const hiprtBuildOptions		buildOptions,
	hiprtScene**				scenesOut );

/** \brief Destroys a scene.
 *
 * This function destroys
 * hiprtScene representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param scene The scene.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyScene( hiprtContext context, hiprtScene scene );

/** \brief Destroys scenes.
 *
 * This function destroys array of
 * hiprtScene representing acceleration structure topology.
 *
 * \param context The HIPRT API context.
 * \param numScenes The number of scenes to be destroyed.
 * \param scenes The scenes.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyScenes( hiprtContext context, uint32_t numScenes, hiprtScene* scenes );

/** \brief Builds or updates a scene.
 *
 * Given a number of hiprtGeometries from the client, this function builds
 * hiprtScene representing top level acceleration structure topology (in case of
 * a build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context The HIPRT API context.
 * \param buildOperation The type of the build operation.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param temporaryBuffer A temporary buffer for build operation.
 * \param stream A stream used for the build.
 * \param sceneOut The resulting scene.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildScene(
	hiprtContext				context,
	hiprtBuildOperation			buildOperation,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	hiprtApiStream				stream,
	hiprtScene					sceneOut );

/** \brief Builds or updates scenes.
 *
 * Given a number of hiprtGeometries from the client, this function builds
 * hiprtScene representing top level acceleration structure topology (in case of
 * a build) or updates acceleration structure keeping topology intact (update).
 *
 * \param context The HIPRT API context.
 * \param buildOperation The type of the build operation.
 * \param numScenes The number of scenes to be built.
 * \param buildInputs An array of build input structs.
 * \param buildOptions Various flags controlling build process.
 * \param temporaryBuffer A temporary buffer for build operation.
 * \param stream A stream used for the build.
 * \param scenesOut The resulting scenes.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildScenes(
	hiprtContext				context,
	hiprtBuildOperation			buildOperation,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInputs,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	hiprtApiStream				stream,
	hiprtScene*					scenesOut );

/** \brief Returns temporary storage requirements for scene build.
 *
 * \param context The HIPRT API context.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param sizeOut The resulting size.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetSceneBuildTemporaryBufferSize(
	hiprtContext context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions, size_t& sizeOut );

/** \brief Returns temporary storage requirements for scene build.
 *
 * \param context The HIPRT API context.
 * \param numScenes The number of scenes to be built.
 * \param buildInput An input parameters for the build.
 * \param buildOptions Various flags controlling build process.
 * \param sizeOut The resulting size.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtGetScenesBuildTemporaryBufferSize(
	hiprtContext				context,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInput,
	const hiprtBuildOptions		buildOptions,
	size_t&						sizeOut );

/** \brief Compacts scene.
 *
 * This function compacts hiprtScene, reducing used memory.
 * The input scene are automatically destroyed.
 *
 * \param context The HIPRT API context.
 * \param stream A stream used for the compaction.
 * \param sceneIn The input scene to be compacted.
 * \param sceneOut The compacted scene.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCompactScene( hiprtContext context, hiprtApiStream stream, hiprtScene sceneIn, hiprtScene& sceneOut );

/** \brief Compacts scenes.
 *
 * This function compacts an array of
 * hiprtScene, reducing used memory.
 * The input scenes are automatically destroyed.
 *
 * \param context The HIPRT API context.
 * \param numScenes The number of scenes to be compacted.
 * \param stream A stream used for the compaction.
 * \param scenesIn The input scenes to be compacted.
 * \param scenesOut The compacted scenes.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCompactScenes(
	hiprtContext context, uint32_t numScenes, hiprtApiStream stream, hiprtScene* scenesIn, hiprtScene** scenesOut );

/** \brief Creates a custom function table.
 *
 * \param context The HIPRT API context.
 * \param numGeomTypes The number of geometry types.
 * \param numRayTypes The number of ray types.
 * \param funcTableOut The resulting table.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtCreateFuncTable( hiprtContext context, uint32_t numGeomTypes, uint32_t numRayTypes, hiprtFuncTable& funcTableOut );

/** \brief Sets an entry in the function table.
 *
 * \param context The HIPRT API context.
 * \param funcTable The function table.
 * \param geomType The geometry type.
 * \param rayType The ray type.
 * \param set The function set to be assigned.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtSetFuncTable( hiprtContext context, hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set );

/** \brief Destroys a custom function table.
 *
 * \param context The HIPRT API context.
 * \param funcTable The function table.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyFuncTable( hiprtContext context, hiprtFuncTable funcTable );

/** \brief Creates a stack buffer (for hiprtGlobalStack or hiprtDynamicStack).
 *
 * \param context The HIPRT API context.
 * \param input The input information needed for the stack buffer allocation.
 * \param stackBufferOut The stack buffer.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtCreateGlobalStackBuffer(
	hiprtContext context, const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut );

/** \brief Destroys a stack buffer.
 *
 * \param context The HIPRT API context.
 * \param stackBuffer The stack buffer.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtDestroyGlobalStackBuffer( hiprtContext context, hiprtGlobalStackBuffer stackBuffer );

/** \brief Saves hiprtGeometry to a binary file.
 *
 * \param context The HIPRT API context.
 * \param geometry The geometry to be saved.
 * \param filename The file name with full path.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtSaveGeometry( hiprtContext context, hiprtGeometry geometry, const char* filename );

/** \brief Loads hiprtGeometry to a binary file.
 *
 * \param context The HIPRT API context.
 * \param geometryOut The geometry to be loaded.
 * \param filename The file name with full path.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtLoadGeometry( hiprtContext context, hiprtGeometry& geometryOut, const char* filename );

/** \brief Saves hiprtScene to a binary file.
 *
 * \param context The HIPRT API context.
 * \param scene The scene to be saved.
 * \param filename The file name with full path.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtSaveScene( hiprtContext context, hiprtScene scene, const char* filename );

/** \brief Loads hiprtScene to a binary file.
 *
 * \param context The HIPRT API context.
 * \param sceneOut The scene to be loaded.
 * \param filename The file name with full path.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtLoadScene( hiprtContext context, hiprtScene& sceneOut, const char* filename );

/** \brief Outputs scene's AABB.
 *
 * \param context The HIPRT API context.
 * \param geometry The geometry to be queried.
 * \param aabbMinOut The bounding box min. bound.
 * \param aabbMaxOut The bounding box max. bound.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtExportGeometryAabb( hiprtContext context, hiprtGeometry geometry, hiprtFloat3& aabbMinOut, hiprtFloat3& aabbMaxOut );

/** \brief Outputs scene AABB.
 *
 * \param context The HIPRT API context.
 * \param scene The scene to be queried.
 * \param aabbMinOut The bounding box min. bound.
 * \param aabbMaxOut The bounding box max. bound.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError
hiprtExportSceneAabb( hiprtContext context, hiprtScene scene, hiprtFloat3& aabbMinOut, hiprtFloat3& aabbMaxOut );

/** \brief Returns function instance with HIPRT routines.
 * \param context The HIPRT API context.
 * \param numFunctions The number of functions to compile.
 * \param funcNames Functions to to be returned, cannot be nullptr.
 * \param src The module source code.
 * \param moduleName The name of the module.
 * \param numHeaders The number of headers.
 * \param headers Sources of the headers, headers can be nullptr when numHeaders is 0.
 * \param includeNames The header names to be included to the module, includeNames can be nullptr
 * when numHeaders is 0.
 * \param numOptions Number of compiler options, can be 0.
 * \param options The compiler options, can be nullptr.
 * \param numGeomTypes The number of geometry types.
 * \param numRayTypes The number of ray types.
 * \param funcNameSets The table of custom function names (numRayTypes x numGeomTypes).
 * \param functionsOut The output function instances.
 * \param moduleOut The output module instance, can be nullptr.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildTraceKernels(
	hiprtContext	  context,
	uint32_t		  numFunctions,
	const char**	  funcNames,
	const char*		  src,
	const char*		  moduleName,
	uint32_t		  numHeaders,
	const char**	  headers,
	const char**	  includeNames,
	uint32_t		  numOptions,
	const char**	  options,
	uint32_t		  numGeomTypes,
	uint32_t		  numRayTypes,
	hiprtFuncNameSet* funcNameSets,
	hiprtApiFunction* functionsOut,
	hiprtApiModule*	  moduleOut,
	bool			  cache );

/** \brief Get function instance with HIPRT routines.
 * \param context The HIPRT API context.
 * \param numFunctions The number of functions to compile.
 * \param funcNames Functions to to be returned, cannot be nullptr.
 * \param moduleName The name of the bitcode module.
 * \param bitcodeBinary The user compiled bitcode.
 * \param bitcodeBinarySize The size of the compiled bitcode.
 * \param numGeomTypes The number of geometry types.
 * \param numRayTypes The number of ray types.
 * \param funcNameSets The table of custom function names (numRayTypes x numGeomTypes).
 * \param functionsOut The output function instances.
 * \return A HIPRT error in case of a failure, hiprtSuccess otherwise.
 */
HIPRT_API hiprtError hiprtBuildTraceKernelsFromBitcode(
	hiprtContext	  context,
	uint32_t		  numFunctions,
	const char**	  funcNames,
	const char*		  moduleName,
	const char*		  bitcodeBinary,
	size_t			  bitcodeBinarySize,
	uint32_t		  numGeomTypes,
	uint32_t		  numRayTypes,
	hiprtFuncNameSet* funcNameSets,
	hiprtApiFunction* functionsOut,
	bool			  cache );

/** \brief Setting log level.
 * \param context The HIPRT API context.
 * \param path A user defined path to cache kernels.
 */
HIPRT_API void hiprtSetCacheDirPath( hiprtContext context, const char* path );

/** \brief Sets the log level.
 *
 * \param level The desired log level.
 */
HIPRT_API void hiprtSetLogLevel( hiprtLogLevel level );

#ifdef __cplusplus
}
#endif
