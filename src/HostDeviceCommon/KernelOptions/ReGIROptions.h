/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_REGIR_OPTIONS_H
#define HOST_DEVICE_COMMON_REGIR_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

#define REGIR_DEBUG_MODE_NO_DEBUG 0
#define REGIR_DEBUG_MODE_GRID_CELLS 1
#define REGIR_DEBUG_MODE_AVERAGE_CELL_NON_CANONICAL_RESERVOIR_CONTRIBUTION 2
#define REGIR_DEBUG_MODE_AVERAGE_CELL_CANONICAL_RESERVOIR_CONTRIBUTION 3
#define REGIR_DEBUG_MODE_REPRESENTATIVE_POINTS 4
#define REGIR_DEBUG_MODE_REPRESENTATIVE_NORMALS 5
#define REGIR_DEBUG_MODE_SAMPLING_FALLBACK 6

#define REGIR_HASH_GRID_COLLISION_RESOLUTION_MODE_LINEAR_PROBING 0
#define REGIR_HASH_GRID_COLLISION_RESOLUTION_MODE_REHASHING 1

 // This block is a security to make sure that we have everything defined otherwise this can lead
 // to weird behavior because of the compiler not knowing about some macros
#ifndef KERNEL_OPTION_TRUE
#error "KERNEL_OPTION_TRUE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#else
#ifndef KERNEL_OPTION_FALSE
#error "KERNEL_OPTION_FALSE not defined, include 'HostDeviceCommon/KernelOptions/Common.h'"
#endif
#endif

/**
 * The resolution downscale factor to apply for the ReGIR grid prepopulation.
 * 
 * The lower the downscale, the more effective the prepoluation but also the more costly
 */
#define ReGIR_GridPrepopulationResolutionDownscale 1

/**
 * Maximum size in bytes of the scratch buffer that is going to be used to compute
 * the contribution of all the lights in the scene to each cell of the ReGIR grid
 * 
 * The bigger the size, the faster the precomputation but obviously the more memory is used
 */
#define ReGIR_ComputeCellsLightDistributionsScratchBufferMaxSizeBytes 200000000u
// This one is just an helper constant computed from the one above and should not be modified directly
#define ReGIR_ComputeCellsLightDistributionsScratchBufferMaxContributionsCount (static_cast<unsigned int>(ReGIR_ComputeCellsLightDistributionsScratchBufferMaxSizeBytes / sizeof(float)))

/**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
* How to sample lights in the scene for filling the ReGIR grid.
*
*	- LSS_BASE_UNIFORM
*		Lights are sampled uniformly
*
*	- LSS_BASE_POWER
*		Lights are sampled proportionally to their power
*
*	- LSS_BASE_LIGHT_TREE_ATS
*		Implementation of [Importance Sampling of Many Lights with Adaptive Tree Splitting, Conty & Kulla, 2018]
*/
#define ReGIR_GridFillLightSamplingBaseStrategy LSS_BASE_LIGHT_TREE_ATS

/**
 * The light sampling strategy used for sampling canonical samples during grid fill
 */
#define ReGIR_GridFillLightSamplingBaseStrategyCanonical LSS_BASE_POWER

/**
 * Whether or not to use a visibility term in the target function used to resample the reservoirs of the grid cells.
 * 
 * Probably too expensive to be efficient.
 */
#define ReGIR_GridFillTargetFunctionVisibility KERNEL_OPTION_FALSE

/**
 * Whether or not to use a the cosine term between the direction to the light sample and the
 * representative normal of the grid cell in the target function used to resample the reservoirs of the grid cells.
 *
 * This has no effect is representative points are not being used
 */
#define ReGIR_GridFillTargetFunctionCosineTerm KERNEL_OPTION_TRUE

/**
 * Takes the cosine term at the light source (i.e. the cosine term of the geometry term) into account when
 * evaluating the target function during grid fill
 */
#define ReGIR_GridFillTargetFunctionCosineTermLightSource KERNEL_OPTION_TRUE

/**
 * Whether or not to include the BSDF in the target function used for the resampling of the initial candidates
 * for the grid fill.
 * 
 * Helps a lot on glossy surfaces.
 * 
 * This option applies to primary hits only and should generally be set to true for better sampling.
 */
#define ReGIR_GridFillPrimaryHitsTargetFunctionBSDF KERNEL_OPTION_TRUE

/**
 * Same as 'ReGIR_GridFillPrimaryHitsTargetFunctionBSDF' but only applies to secondary hits
 * 
 * This option should be set to false in general as we cannot guess in advance what the view direction is going
 * to be at secondary hits (since they can come from anywhere when the rays bounce around the scene) and thus we
 * cannot properly evaluate the BRDF for sampling lights.
 */
#define ReGIR_GridFillSecondaryHitsTargetFunctionBSDF KERNEL_OPTION_FALSE

/**
 * Whether or not to estimate the visibility probability of samples with NEE++ during the grid fill.
 */
#define ReGIR_GridFillTargetFunctionNeePlusPlusVisibilityEstimation KERNEL_OPTION_TRUE

/**
 * This option must be set to true and a grid fill + spatial reuse kernels compiled with this option set
 * to true for those passes to accumulate the RIS integral of the reservoirs (for use in MIS)
 */
#define ReGIR_GridFillSpatialReuse_AccumulatePreIntegration KERNEL_OPTION_FALSE

/**
 * Whether or not to enable light presampling to improve grid fill performance
 * on scenes with many many lights
 */
#define ReGIR_GridFillDoLightPresampling KERNEL_OPTION_FALSE

/**
 * If true, the contribution of each emissive mesh of the scene will be precomputed
 * at each cell of the hash grid to build a sampling distribution based on the contribution
 * of the emissive meshes.
 * 
 * Those per-cell sampling distribution will then be used during the grid fill to provide higher
 * quality initial light samples
 */
#define ReGIR_GridFillUsePerCellLightDistributions KERNEL_OPTION_TRUE

/**
 * If true, ReGIR will not be used to shade points at path tracing time. Only the light distributions precomputed
 * ahead of time will be used to compute NEE
 */
#define ReGIR_ShadingResamplingSampleOnlyLightDistributions KERNEL_OPTION_FALSE

/**
 * How many canonical samples (simple power sampling) to draw and combine with cell-light-distribution
 * samples to guarantee unbiasedness.
 * 
 * 1 guarantees unbiasedness. More than 1 reduces variance more effectively if the coverage of the
 * cell-light-distribution is poor
 */
#define ReGIR_GridFillCellDistributionsCanonicalSampleCount 1

/**
 * Whether or not to use a repsentative normal when computing the contribution of an emissive
 * mesh to the grid cell. This can help quickly reject backfacing lights and should
 * probably be left enabled
 */
#define ReGIR_GridFillCellDistributionsUseRepresentativeNormal KERNEL_OPTION_TRUE

/**
 * When computing the contribution of meshes to the grid cell point:
 * 
 * - If this option is KERNEL_OPTION_TRUE, random points will be chosen on the emissive mesh and the 
 *		contribution to the grid cell point of each of these points on the emissive mesh
 * 		will be integrated to compute an estimate of the overall contribution of the
 *		emissive mesh to the grid cell.
 *		The number of random points drawn is equal to ReGIR_GridFillCellDistributionsIntegrateMeshSampleCount
 * 
 * - If this option is KERNEL_OPTION_FALSE, the overall contribution of the mesh is going to be computed
 *		in one go using an approximate representative point for the whole as well as an average reprensetative
 *		normal. This is less precise than integrating over the mesh but way faster
 */
#define ReGIR_GridFillCellDistributionsIntegrateMesh KERNEL_OPTION_FALSE

/**
 * How many random points to integrate the contribution of an emissive mesh over 
 * if ReGIR_GridFillCellDistributionsIntegrateMesh is KERNEL_OPTION_TRUE
 */
#define ReGIR_GridFillCellDistributionsIntegrateMeshSampleCount 16

/**
 * If this is TRUE, NEE++ visibility estimation will be used in the grid fill target
 * function for non-canonical reservoirs if grid cell light distributions are enabled.
 * 
 * If this is false, NEE++ won't be used in the target function with makes the grid fill
 * quite a bit faster because fetching NEE++ for each non-canonical reservoir is a bit expensive. 
 *
 * With ReGIR spatial reuse enabled (and only if it is enabled) however, this is going to be biased but the bias is
 * actually is very small so this is a worthy optimization imo.
 */
#define ReGIR_GridFillCellDistributionsUnbiasedNEEPlusPlus KERNEL_OPTION_FALSE

/**
 * Whether or not to use a shadow ray in the target function when shading a point at path tracing time.
 * This reduces visibility noise
 */
#define ReGIR_ShadingResamplingTargetFunctionVisibility KERNEL_OPTION_TRUE

/**
 * Whether or not to use NEE++ to estimate the visibility probability of the reservoir being resampled during
 * shading such that reservoirs that are likely to be occluded will have a lower resampling probability
 * 
 * This option is exclusive with ReGIR_ShadingResamplingTargetFunctionVisibility, the latter taking precedence.
 */
#define ReGIR_ShadingResamplingTargetFunctionNeePlusPlusVisibility KERNEL_OPTION_TRUE

/**
 * Whether or not to jitter canonical candidates during the shading resampling.
 * This reduces grid artifacts but increases variance
 */
#define ReGIR_ShadingResamplingJitterCanonicalCandidates KERNEL_OPTION_TRUE

/**
 * Whether or not to include canonical candidates at all during the shading.
 * 
 * Setting this to false is biased but useful basically only for debug purposes
 */
#define ReGIR_ShadingResamplingIncludeCanonicalCandidates KERNEL_OPTION_TRUE

/**
 * Whether or not to incorporate BSDF samples with MIS during shading resampling.
 */
#define ReGIR_ShadingResamplingDoBSDFMIS KERNEL_OPTION_TRUE

/**
 * If this is true, BSDF sample rays will be traced in a BVH that contains only the lights of the scene,
 * not the rest of the geometry. This can increase variance but make the traces way way faster to the point
 * where BSDF MIS rays are almost free
 */
#define ReGIR_ShadingResamplingDoBSDFMISSimplifiedRay KERNEL_OPTION_TRUE

/**
 * If true, all samples resampled will be shaded instead of shading only the reservoir result of the resampling.
 * 
 * This massively improves quality at the cost of performance and is very likely to be worth it for scenes that are not
 * too hard to trace (where shadow rays are expensive).
 */
#define ReGIR_ShadingResamplingShadeAllSamples KERNEL_OPTION_FALSE

/**
 * Light sampling technique used in case the position that we are shading is falling outside of the ReGIR grid
 * 
 * All LSS_BASE_XXX strategies are allowed except LSS_BASE_REGIR
 */
#define ReGIR_FallbackLightSamplingStrategy LSS_BASE_POWER

/**
 * If true, shading point jittering will only jitter the point in the tangent plane of the surface.
 * 
 * This helps reducing bad jittering (jittering which moves the shading point outside of the scene's surface)
 * and reduces variance becaues we're getting more useful neighbors out of the jitters instead of having to rely
 * on jittering-retries to find a valid neighbor
 */
#define ReGIR_JitterInTangentPlane KERNEL_OPTION_TRUE

/**
 * Whether or not to increase the hash grid precision on surfaces that have a lower roughness
 * such that the BRDF term in the target function of the grid fill (if used at all) has a higher
 * precision and gives better results
 */
#define ReGIR_HashGridAdaptiveRoughnessGridPrecision KERNEL_OPTION_TRUE

/**
 * Whether or not to use constant grid cell size for the hash grid.
 * 
 * If this is false, the grid cell size will increase (cells gets bigger) the further away
 * from the camera. This can help with performance and the number of resident cells
 * in the hash grid but it tends to hurt quality because of the reduced grid cell resolution
 */
#define ReGIR_HashGridConstantGridCellSize KERNEL_OPTION_FALSE

/**
 *  How to resolve a collision found in the hash grid:
 * 
 * - REGIR_HASH_GRID_COLLISION_RESOLUTION_LINEAR_PROBING: If a collision is found, look up the next index in the hash
 *      table and see if that location is empty. If not empty, continue looking at the next location
 *      up to 'ReGIR_HashGridCollisionResolutionMaxSteps' times
 * 
 * - REGIR_HASH_GRID_COLLISION_RESOLUTION_REHASHING: If a collision is found, hash the current cell index to get the
 *      new candidate location. Continue doing so until an empty location is found or 'ReGIR_HashGridCollisionResolutionMaxSteps'
 *      steps is exceeded
 */
#define ReGIR_HashGridCollisionResolutionMode REGIR_HASH_GRID_COLLISION_RESOLUTION_MODE_LINEAR_PROBING

/**
 * Maximum number of steps for the linear probing in the hash table to resolve collisions
 */
#define ReGIR_HashGridCollisionResolutionMaxSteps 32

/**
 * If true, cell borders will be randomized a bit to help with grid cell artifacts
 */
#define ReGIR_HashGridHashFuzzyGridCells KERNEL_OPTION_FALSE

/**
 * If true, the normals used in the hash function of the hash grid will be jittered a little bit
 * to help hide grid artifacts caused by the discretization of normals
 */
#define ReGIR_HashGridHashFuzzyNormals KERNEL_OPTION_FALSE

/**
 * Whether or not to use the surface normal in the hash function of the hash grid
 */
#define ReGIR_HashGridHashSurfaceNormal KERNEL_OPTION_TRUE


/** 
 * The number of discretization steps used to hash the surface normal
 * The higher the number, the better the hash grid resolution but the higher the
 * memory cost of the grid and the computational cost of the grid fill
 */
#define ReGIR_HashGridHashSurfaceNormalResolutionPrimaryHits 4

/**
 * Same as above but for the secondary hits only. A lower setting here is usually enough and saves
 * on perf and VRAM
 */
#define ReGIR_HashGridHashSurfaceNormalResolutionSecondaryHits 2

/**
 * If using jittering, how many retries to perform to find a good neighbor at shading time?
 *
 * This is because with jittering, our jittered position may end up outside of the grid
 * or in an empty cell, in which case we want to retry with a differently jittered position
 * to try and find a good neighbor
 */
#define ReGIR_ShadingJitterRetries 2

/**
 * Debug option to color the scene with the grid cells
 */
#define ReGIR_DebugMode REGIR_DEBUG_MODE_NO_DEBUG
//#define ReGIR_DebugMode REGIR_DEBUG_MODE_GRID_CELLS

#endif // #ifndef __KERNELCC__

#endif
