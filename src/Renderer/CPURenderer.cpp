/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/BSDFs/LTCsData/GGXConductorLTCFitData.h"

#include "Device/kernels/CameraRays.h"
#include "Device/kernels/Megakernel.h"
#include "Device/kernels/Neural/NISMLGridOptimize.h"
#include "Device/kernels/Neural/NISMLOptimize.h"
#include "Device/kernels/Neural/NISMLTrain.h"

#include "Device/kernels/NEE++/GridPrepopulate.h"
#include "Device/kernels/NEE++/NEEPlusPlusFinalizeAccumulation.h"

#include "Device/kernels/ReSTIR/DirectionalReuseCompute.h"

#include "Device/kernels/ReSTIR/GI/InitialCandidates.h"
#include "Device/kernels/ReSTIR/GI/Shading.h"
#include "Device/kernels/ReSTIR/GI/SpatialReuse.h"
#include "Device/kernels/ReSTIR/GI/TemporalReuse.h"

#include "Device/kernels/ReSTIR/PT/InitialCandidates.h"
#include "Device/kernels/ReSTIR/PT/Shading.h"
#include "Device/kernels/ReSTIR/PT/SpatialReuse.h"
#include "Device/kernels/ReSTIR/PT/SpatialReuseSPMIS.h"
#include "Device/kernels/ReSTIR/PT/TemporalReuse.h"

#include "Device/kernels/ReSTIR/SPMIS/BuildCDFs.h"
#include "Device/kernels/ReSTIR/SPMIS/ComputeCellsVariance.h"
#include "Device/kernels/ReSTIR/SPMIS/ComputeOffsets.h"
#include "Device/kernels/ReSTIR/SPMIS/CountCells.h"
#include "Device/kernels/ReSTIR/SPMIS/ResetBuffers.h"
#include "Device/kernels/ReSTIR/SPMIS/ResetCellsData.h"
#include "Device/kernels/ReSTIR/SPMIS/Sort.h"

#include "Device/kernels/ReSTIR/PG/Fitting.h"
#include "Device/kernels/ReSTIR/PG/ResetDistributions.h"
#include "Device/kernels/ReSTIR/PG/ResetHashGrid.h"
#include "Device/kernels/ReSTIR/PG/ResetSufficientStatistics.h"
#include "Device/kernels/ReSTIR/PG/Splatting.h"

#include "Device/kernels/GMoN/GMoNComputeMedianOfMeans.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreAccumulateBatchStatisticsIntoHistory.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreAccumulateBatchTrainingSamples.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreExpandOneLookaheadLevel.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreInitializeCreatedNodeHistory.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreMarkGuidingCellsForSplitting.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CorePromoteGuidingCells.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreReduceBatchStatisticsUpward.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreReplayTrainingSamples.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreResetBatchKDTreeStatistics.h"
#include "Device/kernels/IlluminationAwareKDTree/Core/CoreResetTree.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterAccumulateNormalFaceObservations.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterAllocateNormalFaceLightcuts.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterBuildLightcutSamplingCDFs.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterInitializeLightcutQ0.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterInitializeRootLightcut.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterInitializeShadingContexts.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterQUpdates.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterRefineLightcuts.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterReplayQRewards.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterReplayStatistics.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterResetBatchKDTreeAndLightcutStatistics.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterResetBatchLightcutStatistics.h"
#include "Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterStatisticsUpdates.h"
#include "Device/kernels/IlluminationAwareKDTree/NISML/NISMLBuildCaches.h"
#include "Device/kernels/IlluminationAwareKDTree/NISML/NISMLReplayTrainingSamples.h"
#include "Device/kernels/SSBNPermutation/SortingPass.h"

#include "Renderer/Baker/GPUBaker.h"
#include "Renderer/Baker/GPUBakerConstants.h"
#include "Renderer/CPURenderer.h"

#include "Device/kernels/HierarchicalAdaptiveSampling/BuildColumns.h"
#include "Device/kernels/HierarchicalAdaptiveSampling/BuildHierarchy.h"
#include "Device/kernels/HierarchicalAdaptiveSampling/BuildRows.h"
#include "Device/kernels/HierarchicalAdaptiveSampling/ComputeError.h"
#include "Device/kernels/HierarchicalAdaptiveSampling/ResolveMask.h"
#include "Threads/ThreadManager.h"
#include "UI/ApplicationSettings.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <numeric>
#include <omp.h>
#include <ranges>

// If 1, only the pixel at DEBUG_PIXEL_X and DEBUG_PIXEL_Y will be rendered,
// allowing for fast step into that pixel with the debugger to see what's happening.
// Otherwise if 0, all pixels of the image are rendered
#define DEBUG_PIXEL 0

// If 0, the pixel with coordinates (x, y) = (0, 0) is top left corner (image viewer).
// If 1, it's bottom left corner (this renderer).
//
// Useful if you're using an image viewer to get the the coordinates of the interesting pixel. If that image viewer has its (0, 0) in the top left corner,
// you'll need to set that DEBUG_FLIP_Y to 0. Set 1 to if you're measuring the coordinates of the pixel with (0, 0) in the bottom left corner
//
// If you're debugging coordinates that come from a kernel of this renderer (by printing the x and y coordinate of the pixel being rendered, then this will need
// to be 1)
#define DEBUG_FLIP_Y 0

// Coordinates of the pixel whose neighborhood needs to rendered (useful for algorithms
// where pixels are not completely independent from each other such as ReSTIR Spatial Reuse).
//
// The neighborhood around pixel will be rendered if DEBUG_RENDER_NEIGHBORHOOD is 1.
#define DEBUG_PIXEL_X 109
#define DEBUG_PIXEL_Y 183

// Same as DEBUG_FLIP_Y but for the "other debug pixel"
#define DEBUG_OTHER_FLIP_Y 0

// Allows to render the neighborhood around the DEBUG_PIXEL_X/Y but to debug at the location
// of DEBUG_OTHER_PIXEL_X/Y given below.
//
// -1 to disable. If disabled, the pixel at (DEBUG_PIXEL_X, DEBUG_PIXEL_Y) will be debugged
#define DEBUG_OTHER_PIXEL_X -1
#define DEBUG_OTHER_PIXEL_Y -1

// If 1, a square of DEBUG_NEIGHBORHOOD_SIZE x DEBUG_NEIGHBORHOOD_SIZE pixels
// will be rendered around the pixel to debug (given by DEBUG_PIXEL_X and
// DEBUG_PIXEL_Y). The pixel of interest is going to be rendered first so you
// can just set a breakpoint in the pass of interest and it will break when rendering the
// pixel that you want to debug.
// This can be useful when debugging spatial passes such as ReSTIR spatial reusing.
// If you were only rendering the precise pixel at the given debug coordinates, you
// wouldn't be able to debug correctly since all the neighborhood wouldn't have been
// rendered which means no reservoir which means improper rendering
#define DEBUG_RENDER_NEIGHBORHOOD 1
// How many pixels to render around the debugged pixel given by the DEBUG_PIXEL_X and
// DEBUG_PIXEL_Y coordinates
#define DEBUG_NEIGHBORHOOD_SIZE 150

CPURenderer::CPURenderer(int width, int height) : m_resolution(make_int2(width, height))
{
	m_render_data.render_settings.render_resolution = m_resolution;
}

void CPURenderer::setup_buffers()
{
	unsigned int width	= m_resolution.x;
	unsigned int height = m_resolution.y;

	m_framebuffer = Image32Bit(width, height, 3);
	m_last_frame_ray_colors.resize(width * height);

	// Resizing buffers + initial value
	m_pixel_active_buffer.resize(width * height, 0);
	m_input_random_seeds.resize(width * height, 0);
	m_updated_random_seeds.resize(width * height, 0);
	m_denoiser_albedo.resize(width * height, ColorRGB32F(0.0f));
	m_denoiser_normals.resize(width * height, float3_t{ 0.0f, 0.0f, 0.0f });
	m_pixel_sample_count.resize(width * height, 0);
	m_pixel_converged_sample_count.resize(width * height, 0);
	m_pixel_luminance.resize(width * height, 0.0f);
	m_pixel_squared_luminance.resize(width * height, 0.0f);
	m_hierarchical_adaptive_sampling_error.resize(width * height, 0.0f);
	m_hierarchical_adaptive_sampling_summed_area.resize(width * height, 0.0f);
	unsigned int pixel_count			   = width * height;
	unsigned int maximum_useful_node_count = pixel_count > 0 ? pixel_count * 2u - 1u : 1u;
	unsigned int requested_node_count = static_cast<unsigned int>(std::max(1, m_render_data.render_settings.hierarchical_adaptive_sampling_max_node_count));
	m_hierarchical_adaptive_sampling_nodes.resize(std::min(maximum_useful_node_count, requested_node_count));

#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
	m_restir_gi_state.initial_candidates_reservoirs.resize(width * height);
	m_restir_gi_state.temporal_reservoirs.resize(width * height);
	m_restir_gi_state.spatial_reservoirs.resize(width * height);
	m_restir_gi_state.directional_spatial_reuse_data_buffer.resize(width, height);
#elif PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT // #if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
	m_restir_pt_state.initial_candidates_reservoirs.resize(width * height);
	m_restir_pt_state.temporal_reservoirs.resize(width * height);
	m_restir_pt_state.spatial_reservoirs.resize(width * height);

	m_restir_pt_state.directional_spatial_reuse_data_buffer.resize(width, height);
	m_restir_pt_state.spmis_data.resize(width, height);
	m_restir_pt_state.spmis_data.m_spmis_data.memset_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_ALL_PIXEL_HASHES_CHECKSUMS>(
		HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);
#endif												  // #if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
	m_restir_pg_state.splatting_samples_soa_buffer.resize(width, height, m_render_data.render_settings.nb_bounces);

	m_restir_pg_state.hash_grid_distributions_soa_buffer.resize(ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT, ReSTIRPGDistributionComponentCount);
	m_restir_pg_state.hash_grid_checksums = std::vector<AtomicType<unsigned int>>(ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT);

	m_restir_pg_state.hash_grid_distributions_sufficient_statistics_soa_buffer.resize(ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT,
																					  ReSTIRPGDistributionComponentCount);

	m_restir_pg_state.grid_cell_alive = std::vector<AtomicType<unsigned int>>(ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT);
	m_restir_pg_state.grid_cell_alive_list.resize(ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT);
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE

	m_g_buffer.resize(width * height);
	m_g_buffer_prev_frame.resize(width * height);

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_nisml_state.m_mlp.resize();
	m_nisml_state.m_mlp.initialize(true);
	m_nisml_state.m_position_learnable_dense_grid.resize();
	m_nisml_state.m_position_learnable_dense_grid.initialize();
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.resize(
		IlluminationAwareKDTreeCoreDataHost<std::vector>::MAXIMUM_NUMBER_OF_NODES,
		IlluminationAwareKDTreeCoreDataHost<std::vector>::INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY);
#endif

	setup_bsdfs_data();
	setup_nee_plus_plus();
	setup_gmon();
}

void CPURenderer::setup_bsdfs_data()
{
	m_bsdf_data_cpu_data.load_bsdf_data(m_render_data);
}

void CPURenderer::setup_nee_plus_plus()
{
#if DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
	// Only doing if using NEE++

	m_nee_plus_plus.total_num_rays			= std::vector<AtomicType<unsigned char>>(1000000);
	m_nee_plus_plus.total_unoccluded_rays	= std::vector<AtomicType<unsigned char>>(1000000);
	m_nee_plus_plus.num_rays_staging		= std::vector<AtomicType<unsigned int>>(1000000);
	m_nee_plus_plus.unoccluded_rays_staging = std::vector<AtomicType<unsigned int>>(1000000);
	m_nee_plus_plus.checksum_buffer			= std::vector<AtomicType<unsigned int>>(1000000);
	for (AtomicType<unsigned int>& checksum : m_nee_plus_plus.checksum_buffer)
		checksum.store(HashGrid::UNDEFINED_CHECKSUM_OR_GRID_INDEX);

	m_render_data.nee_plus_plus.m_entries_buffer.total_num_rays		   = m_nee_plus_plus.total_num_rays.data();
	m_render_data.nee_plus_plus.m_entries_buffer.total_unoccluded_rays = m_nee_plus_plus.total_unoccluded_rays.data();
	// m_render_data.nee_plus_plus.m_entries_buffer.num_rays_staging = m_nee_plus_plus.num_rays_staging.data();
	// m_render_data.nee_plus_plus.m_entries_buffer.unoccluded_rays_staging = m_nee_plus_plus.unoccluded_rays_staging.data();
	m_render_data.nee_plus_plus.m_entries_buffer.checksum_buffer = m_nee_plus_plus.checksum_buffer.data();

	m_render_data.nee_plus_plus.m_total_number_of_cells = 1000000;

	m_render_data.nee_plus_plus.m_total_cells_alive_count = &m_nee_plus_plus.total_cell_alive_count;
#endif // #if DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE
}

void CPURenderer::setup_gmon()
{
	if (m_render_data.render_settings.samples_per_frame < m_gmon.number_of_sets)
		m_render_data.buffers.gmon_estimator.use_gmon = false;

	if (m_render_data.buffers.gmon_estimator.use_gmon)
	{
		m_gmon.resize(m_resolution.x, m_resolution.y);
		m_render_data.buffers.gmon_estimator.sets				= m_gmon.sets.data();
		m_render_data.buffers.gmon_estimator.result_framebuffer = m_gmon.result_framebuffer.get_data_as_ColorRGB32F();
	}
}

void CPURenderer::GMoN_post_sample_update()
{
	if (m_render_data.buffers.gmon_estimator.use_gmon)
	{
		m_render_data.buffers.gmon_estimator.next_set_to_accumulate++;

		if (m_render_data.buffers.gmon_estimator.next_set_to_accumulate % m_gmon.number_of_sets == 0)
		{
			// We've added 1 sample to each sets of GMoN so we can compute the median of means
			gmon_compute_median_of_means();

			m_render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;
		}
	}
}

void CPURenderer::ReSTIR_PT_post_sample_update()
{
#if PathSamplingStrategy != PATH_SAMPLING_RESTIR_PT
	return;
#endif

	unsigned int pixel_count = m_resolution.x * m_resolution.y;

#pragma omp parallel for
	for (int index = 0; index < pixel_count; index++)
	{
		ReSTIR_SPMIS_ResetBuffers(m_render_data, index);
	}
}

void CPURenderer::set_scene(Scene& parsed_scene)
{
	m_render_data.GPU_BVH = nullptr;
	m_scene_bounding_box  = parsed_scene.metadata.scene_bounding_box;

	std::vector<DevicePackedTexturedMaterial> gpu_packed_materials;
	gpu_packed_materials.resize(parsed_scene.materials.size());
	for (int i = 0; i < parsed_scene.materials.size(); i++)
		gpu_packed_materials[i] = parsed_scene.materials[i].pack_to_GPU();

	m_gpu_packed_materials.upload_data(gpu_packed_materials);
	m_render_data.buffers.materials_buffer_soa				  = m_gpu_packed_materials.get_device_SoA_struct();
	m_render_data.buffers.material_indices					  = parsed_scene.material_indices.data();
	m_render_data.buffers.has_vertex_normals				  = parsed_scene.has_vertex_normals.data();
	m_render_data.buffers.triangles_indices					  = parsed_scene.triangles_vertex_indices.data();
	m_render_data.buffers.global_triangle_index_to_mesh_index = parsed_scene.global_triangle_index_to_mesh_index.data();
	m_render_data.buffers.vertices_positions				  = parsed_scene.vertices_positions.data();
	m_render_data.buffers.vertex_normals					  = parsed_scene.vertex_normals.data();
	m_render_data.buffers.texcoords							  = parsed_scene.texcoords.data();

	ThreadManager::join_threads(ThreadManager::SCENE_LOADING_COMPUTE_TRIANGLE_AREAS);
	m_render_data.buffers.triangles_areas = parsed_scene.triangle_areas.data();

	ThreadManager::join_threads(ThreadManager::SCENE_TEXTURES_LOADING_THREAD_KEY);
	m_render_data.buffers.material_textures = parsed_scene.textures.data();

	// Computing the opaqueness of materials i.e. whether or not they are FULLY opaque
	m_material_opaque.resize(parsed_scene.materials.size());
	for (int i = 0; i < parsed_scene.materials.size(); i++)
		m_material_opaque[i] = parsed_scene.material_has_opaque_base_color_texture[i] && parsed_scene.materials[i].alpha_opacity == 1.0f;

	m_render_data.buffers.material_opaque = m_material_opaque.data();

	ThreadManager::join_threads(ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	m_render_data.buffers.triangles_average_emissive_luminance		 = parsed_scene.triangles_average_emissive_luminance.data();
	m_render_data.buffers.triangles_average_emissive_power_luminance = parsed_scene.triangles_average_emissive_power_luminance.data();

	m_render_data.buffers.emissive_triangles_count			   = parsed_scene.emissive_triangles_primitive_indices.size();
	m_render_data.buffers.emissive_triangles_primitive_indices = parsed_scene.emissive_triangles_primitive_indices.data();
	m_render_data.buffers.emissive_triangles_primitive_indices_and_emissive_textures =
		parsed_scene.emissive_triangles_primitive_indices_and_emissive_textures.data();

	m_emissive_meshes_alias_tables.load_from_emissive_meshes(parsed_scene);

	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Building scene's BVH...");
	m_triangle_buffer			= parsed_scene.get_triangles(parsed_scene.triangles_vertex_indices);
	m_emissive_triangles_buffer = parsed_scene.get_triangles(parsed_scene.emissive_triangle_vertex_indices);

	m_bvh		= std::make_shared<BVH>(&m_triangle_buffer);
	m_light_bvh = std::make_shared<BVH>(&m_emissive_triangles_buffer);

#if DirectLightSamplingStrategy == LSS_BASE_POWER
	g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_INFO, "Building scene's power alias table");
	compute_emissives_power_alias_table(parsed_scene);
#endif // #if DirectLightSamplingStrategy == LSS_BASE_POWER

#if DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS ||                                                                                                  \
	(DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE && NEEPlusPlusGridPrepopulateLightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
	m_light_tree_builder_ats.build_light_tree(parsed_scene.emissive_triangles_primitive_indices, parsed_scene.triangles_average_emissive_power_luminance,
											  parsed_scene.triangles_vertex_indices, parsed_scene.vertices_positions);
	m_light_tree_ats_build_result = m_light_tree_builder_ats.compute_build_result<std::vector>();
	m_light_tree_builder_ats.to_device(m_render_data, parsed_scene.emissive_triangles_primitive_indices, parsed_scene.triangles_vertex_indices.size() / 3,
									   m_light_tree_ats_build_result);
	m_light_tree_builder_ats.cleanup();
#endif // #if DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS || (DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE &&
	   // NEEPlusPlusGridPrepopulateLightSamplingStrategy == LSS_BASE_LIGHT_TREE_ATS)
#if DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG ||                                                                                                   \
	(DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE && NEEPlusPlusGridPrepopulateLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)
	if (parsed_scene.emissive_triangles_primitive_indices.size() > 0)
	{
		m_light_tree_builder_sg.build_light_tree(parsed_scene.emissive_triangles_primitive_indices, parsed_scene.triangles_average_emissive_power_luminance,
												 parsed_scene.triangles_vertex_indices, parsed_scene.vertices_positions,
												 static_cast<unsigned int>(parsed_scene.triangles_vertex_indices.size() / 3));
		m_light_tree_sg_build_result = m_light_tree_builder_sg.compute_build_result<std::vector>();
		m_light_tree_builder_sg.to_device(m_render_data, parsed_scene.emissive_triangles_primitive_indices, parsed_scene.triangles_vertex_indices.size() / 3,
										  m_light_tree_sg_build_result);
		m_light_tree_builder_sg.get_nisml_data().to_device<std::vector>(m_render_data.nisml);
		m_light_tree_builder_sg.cleanup();
	}
#endif // #if DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG || (DirectLightUseNEEPlusPlus == KERNEL_OPTION_TRUE &&
	   // NEEPlusPlusGridPrepopulateLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_nisml_state.m_nisml_data.resize();
#endif
}

void CPURenderer::update_render_data()
{
	bsdfs_data_to_device();

	m_render_data.buffers.accumulated_ray_colors						   = m_framebuffer.get_data_as_ColorRGB32F();
	m_render_data.buffers.last_frame_ray_colors							   = m_last_frame_ray_colors.data();
	m_render_data.aux_buffers.pixel_active								   = m_pixel_active_buffer.data();
	m_render_data.aux_buffers.denoiser_albedo							   = m_denoiser_albedo.data();
	m_render_data.aux_buffers.denoiser_normals							   = m_denoiser_normals.data();
	m_render_data.aux_buffers.pixel_sample_count						   = m_pixel_sample_count.data();
	m_render_data.aux_buffers.pixel_converged_sample_count				   = m_pixel_converged_sample_count.data();
	m_render_data.aux_buffers.pixel_luminance							   = m_pixel_luminance.data();
	m_render_data.aux_buffers.pixel_squared_luminance					   = m_pixel_squared_luminance.data();
	m_render_data.aux_buffers.hierarchical_adaptive_sampling_error		   = m_hierarchical_adaptive_sampling_error.data();
	m_render_data.aux_buffers.hierarchical_adaptive_sampling_summed_area   = m_hierarchical_adaptive_sampling_summed_area.data();
	m_render_data.aux_buffers.hierarchical_adaptive_sampling_nodes		   = m_hierarchical_adaptive_sampling_nodes.data();
	m_render_data.aux_buffers.hierarchical_adaptive_sampling_node_count	   = &m_hierarchical_adaptive_sampling_node_count;
	m_render_data.aux_buffers.hierarchical_adaptive_sampling_node_capacity = static_cast<unsigned int>(m_hierarchical_adaptive_sampling_nodes.size());
	m_render_data.aux_buffers.still_one_ray_active						   = &m_still_one_ray_active;
	m_render_data.aux_buffers.pixel_count_converged_so_far				   = &m_stop_noise_threshold_count;
	m_render_data.buffers.set_input_random_seed_pointer(m_input_random_seeds.data());
	m_render_data.buffers.set_updated_random_seed_pointer(m_updated_random_seeds.data());

	m_render_data.g_buffer.materials			= m_g_buffer.materials.data();
	m_render_data.g_buffer.geometric_normals	= m_g_buffer.geometric_normals.data();
	m_render_data.g_buffer.shading_normals		= m_g_buffer.shading_normals.data();
	m_render_data.g_buffer.primary_hit_position = m_g_buffer.primary_hit_position.data();
	m_render_data.g_buffer.first_hit_prim_index = m_g_buffer.first_hit_prim_index.data();

	m_render_data.g_buffer_prev_frame.materials			   = m_g_buffer_prev_frame.materials.data();
	m_render_data.g_buffer_prev_frame.geometric_normals	   = m_g_buffer_prev_frame.geometric_normals.data();
	m_render_data.g_buffer_prev_frame.shading_normals	   = m_g_buffer_prev_frame.shading_normals.data();
	m_render_data.g_buffer_prev_frame.primary_hit_position = m_g_buffer_prev_frame.primary_hit_position.data();
	m_render_data.g_buffer_prev_frame.first_hit_prim_index = m_g_buffer_prev_frame.first_hit_prim_index.data();

#if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
	m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_restir_gi_state.initial_candidates_reservoirs.data();
	m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs				  = m_restir_gi_state.initial_candidates_reservoirs.data();
	m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs			  = m_restir_gi_state.temporal_reservoirs.data();
	m_render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs				  = m_restir_gi_state.temporal_reservoirs.data();
	m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs				  = m_restir_gi_state.spatial_reservoirs.data();
	m_render_data.aux_buffers.restir_gi_reservoir_buffer_1										  = m_restir_gi_state.initial_candidates_reservoirs.data();
	m_render_data.aux_buffers.restir_gi_reservoir_buffer_2										  = m_restir_gi_state.spatial_reservoirs.data();
	m_render_data.aux_buffers.restir_gi_reservoir_buffer_3										  = m_restir_gi_state.temporal_reservoirs.data();
	m_restir_gi_state.directional_spatial_reuse_data_buffer.to_device<ReSTIR_VARIANT_GI>(m_render_data);
#elif PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT // #if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
	m_render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer = m_restir_pt_state.initial_candidates_reservoirs.data();
	m_render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs				  = m_restir_pt_state.initial_candidates_reservoirs.data();
	m_render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs			  = m_restir_pt_state.temporal_reservoirs.data();
	m_render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs				  = m_restir_pt_state.temporal_reservoirs.data();
	m_render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs				  = m_restir_pt_state.spatial_reservoirs.data();

	m_render_data.aux_buffers.restir_pt_reservoir_buffer_1 = m_restir_pt_state.initial_candidates_reservoirs.data();
	m_render_data.aux_buffers.restir_pt_reservoir_buffer_2 = m_restir_pt_state.spatial_reservoirs.data();
	m_render_data.aux_buffers.restir_pt_reservoir_buffer_3 = m_restir_pt_state.temporal_reservoirs.data();

	m_restir_pt_state.directional_spatial_reuse_data_buffer.template to_device<ReSTIR_VARIANT_PT>(m_render_data);
	m_restir_pt_state.spmis_data.to_device(m_render_data);
#endif												  // #if PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
	m_render_data.render_settings.restir_pg_settings.splatting_samples_soa = m_restir_pg_state.splatting_samples_soa_buffer.to_device();

	m_render_data.render_settings.restir_pg_settings.hash_grid_distributions_soa = m_restir_pg_state.hash_grid_distributions_soa_buffer.to_device();
	m_render_data.render_settings.restir_pg_settings.hash_grid_checksums		 = m_restir_pg_state.hash_grid_checksums.data();
	m_render_data.render_settings.restir_pg_settings.grid_cell_alive			 = m_restir_pg_state.grid_cell_alive.data();
	m_render_data.render_settings.restir_pg_settings.grid_cell_alive_count		 = &m_restir_pg_state.grid_cell_alive_count;
	m_render_data.render_settings.restir_pg_settings.grid_cell_alive_list		 = m_restir_pg_state.grid_cell_alive_list.data();

	m_render_data.render_settings.restir_pg_settings.hash_grid_distributions_sufficient_statistics_soa =
		m_restir_pg_state.hash_grid_distributions_sufficient_statistics_soa_buffer.to_device();

	m_render_data.render_settings.restir_pg_settings.hash_grid_total_number_of_cells = ReSTIRPGRenderPass::HASH_GRID_INITIAL_CELL_COUNT;
#endif // #if ReSTIRPGEnable == KERNEL_OPTION_TRUE

	m_render_data.buffers.emissive_meshes_data = m_emissive_meshes_alias_tables.to_device();

	m_render_data.cpu_only.bvh		 = m_bvh.get();
	m_render_data.cpu_only.light_bvh = m_light_bvh.get();

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_render_data.nisml.mlp							  = m_nisml_state.m_mlp.to_device(m_nisml_state.m_adam_learning_rate);
	m_render_data.nisml.position_learnable_dense_grid = m_nisml_state.m_position_learnable_dense_grid.to_device(m_nisml_state.m_adam_learning_rate);
	NISMLDevice training_data						  = m_nisml_state.m_nisml_data.to_device();
	m_render_data.nisml.training_records			  = training_data.training_records;
	m_render_data.nisml.training_record_count		  = training_data.training_record_count;
	m_render_data.nisml.training_record_capacity	  = training_data.training_record_capacity;
	m_render_data.nisml.learning_enabled =
		m_nisml_state.m_training_spp <= 0 || m_render_data.render_settings.sample_number < static_cast<unsigned int>(m_nisml_state.m_training_spp);
	m_render_data.nisml.training_record_probability = std::clamp(m_nisml_state.m_training_record_percentage / 100.0f, 0.0f, 1.0f);
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_render_data.kd_tree_device = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.to_device(m_render_data);
#else
	m_render_data.kd_tree_device = {};
#endif
}

void CPURenderer::bsdfs_data_to_device()
{
	m_bsdf_data_cpu_data.to_device(m_render_data);
}

void CPURenderer::compute_emissives_power_alias_table(const Scene& scene)
{
	ThreadManager::add_dependency(ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE, ThreadManager::SCENE_LOADING_PARSE_EMISSIVE_TRIANGLES);
	ThreadManager::start_thread(
		ThreadManager::RENDERER_COMPUTE_EMISSIVES_POWER_ALIAS_TABLE,
		[this, &scene]()
		{
			auto start = std::chrono::high_resolution_clock::now();

			std::vector<float> power_list(scene.emissive_triangles_primitive_indices.size());
			float power_sum = 0.0f;

			for (int emissive_triangle_index = 0; emissive_triangle_index < scene.emissive_triangles_primitive_indices.size(); emissive_triangle_index++)
			{
				int emissive_triangle_global_index = scene.emissive_triangles_primitive_indices[emissive_triangle_index];

				float power = scene.triangles_average_emissive_power_luminance[emissive_triangle_global_index];

				power_list[emissive_triangle_index] = power;
				power_sum += power;
			}

			Utils::compute_alias_table(power_list, power_sum, m_power_alias_table_probas, m_power_alias_table_alias);

			m_render_data.buffers.emissive_triangles_power_alias_table.alias_table_alias  = m_power_alias_table_alias.data();
			m_render_data.buffers.emissive_triangles_power_alias_table.alias_table_probas = m_power_alias_table_probas.data();
			m_render_data.buffers.emissive_triangles_power_alias_table.sum_elements		  = power_sum;
			m_render_data.buffers.emissive_triangles_power_alias_table.size				  = scene.emissive_triangles_primitive_indices.size();

			auto stop = std::chrono::high_resolution_clock::now();
			std::cout << "Power alias table construction time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms"
					  << std::endl;
		});
}

void CPURenderer::set_envmap(Image32Bit& envmap_image)
{
	ThreadManager::join_threads(ThreadManager::ENVMAP_LOAD_FROM_DISK_THREAD);

	if (envmap_image.width == 0 || envmap_image.height == 0)
	{
		m_render_data.world_settings.ambient_light_type = AmbientLightType::UNIFORM;

		std::cout << "Empty envmap set on the CPURenderer... Defaulting to uniform ambient light type" << std::endl;

		return;
	}

	if (EnvmapSamplingStrategy == ESS_BINARY_SEARCH)
	{
		m_envmap_cdf								  = envmap_image.compute_cdf();
		m_render_data.world_settings.envmap_total_sum = m_envmap_cdf.back();
	}
	else if (EnvmapSamplingStrategy == ESS_ALIAS_TABLE)
	{
		float total_sum;

		envmap_image.compute_alias_table(m_envmap_alias_table_probas, m_envmap_alias_table_alias, &total_sum);
		m_render_data.world_settings.envmap_total_sum = total_sum;
	}

	float envmap_scaling_factor;
	m_packed_envmap.pack_from(envmap_image, envmap_scaling_factor);
	m_render_data.world_settings.envmap						  = m_packed_envmap.get_data_pointer();
	m_render_data.world_settings.envmap_width				  = envmap_image.width;
	m_render_data.world_settings.envmap_height				  = envmap_image.height;
	m_render_data.world_settings.envmap_packed_scaling_factor = envmap_scaling_factor;
	m_render_data.world_settings.ambient_light_type			  = AmbientLightType::ENVMAP;

	if (EnvmapSamplingStrategy == ESS_BINARY_SEARCH)
		m_render_data.world_settings.envmap_cdf = m_envmap_cdf.data();
	else if (EnvmapSamplingStrategy == ESS_ALIAS_TABLE)
	{
		m_render_data.world_settings.envmap_alias_table.alias_table_probas = m_envmap_alias_table_probas.data();
		m_render_data.world_settings.envmap_alias_table.alias_table_alias  = m_envmap_alias_table_alias.data();
		m_render_data.world_settings.envmap_alias_table.sum_elements	   = m_render_data.world_settings.envmap_total_sum;
		m_render_data.world_settings.envmap_alias_table.size			   = envmap_image.width * envmap_image.height;
	}
}

void CPURenderer::set_camera(Camera& camera)
{
	m_camera					 = camera;
	m_render_data.current_camera = camera.to_hiprt(m_resolution.x, m_resolution.y);
}

HIPRTRenderData& CPURenderer::get_render_data()
{

	return m_render_data;
}

HIPRTRenderSettings& CPURenderer::get_render_settings()
{

	return m_render_data.render_settings;
}

Image32Bit& CPURenderer::get_framebuffer()
{
	if (m_render_data.buffers.gmon_estimator.use_gmon)

		return m_gmon.result_framebuffer;
	else

		return m_framebuffer;
}

void CPURenderer::render()
{
	std::cout << "CPU rendering..." << std::endl;

	auto start = std::chrono::high_resolution_clock::now();

	nee_plus_plus_cache_visibility_pass();
#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
	ReSTIR_PG_reset_hash_grid();
	ReSTIR_PG_reset_distributions();
#endif

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS ||                                                                                                       \
	(DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) && DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG)
	illumination_aware_kd_tree_reset();
#endif
	// LSS_BASE_LIGHT_TREE_SG)

	// Using 'samples_per_frame' as the number of samples to render on the CPU
	for (int frame_number = 1; frame_number <= m_render_data.render_settings.samples_per_frame; frame_number++)
	{
		m_render_data.render_settings.do_update_status_buffers = true;

		pre_frame_render_update(frame_number);
		update_cameras(frame_number);

		camera_rays_pass();

#if PathSamplingStrategy == PATH_SAMPLING_BSDF || PathSamplingStrategy == PATH_SAMPLING_BSDF_WAVEFRONT
		tracing_pass();
#elif PathSamplingStrategy == PATH_SAMPLING_RESTIR_GI
		ReSTIR_GI_pass();
#elif PathSamplingStrategy == PATH_SAMPLING_RESTIR_PT
		ReSTIR_PT_pass();
#endif // #if PathSamplingStrategy == PATH_SAMPLING_BSDF || PathSamplingStrategy == PATH_SAMPLING_BSDF_WAVEFRONT

#if ReSTIRPGEnable == KERNEL_OPTION_TRUE
		ReSTIR_PG_pass();
#endif

		hierarchical_adaptive_sampling_pass();

		post_sample_update(frame_number);

		std::cout << "Frame " << frame_number << ": " << frame_number / static_cast<float>(m_render_data.render_settings.samples_per_frame) * 100.0f << "%"
				  << std::endl;
	}

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
}

void CPURenderer::pre_frame_render_update(int frame_number)
{
#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_render_data.nisml.learning_enabled =
		m_nisml_state.m_training_spp <= 0 || m_render_data.render_settings.sample_number < static_cast<unsigned int>(m_nisml_state.m_training_spp);
	m_render_data.nisml.training_record_probability = std::clamp(m_nisml_state.m_training_record_percentage / 100.0f, 0.0f, 1.0f);

	NISMLDevice training_data					 = m_nisml_state.m_nisml_data.to_device();
	m_render_data.nisml.training_records		 = training_data.training_records;
	m_render_data.nisml.training_record_count	 = training_data.training_record_count;
	m_render_data.nisml.training_record_capacity = training_data.training_record_capacity;
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.to_device(m_render_data);
	unsigned int node_count						 = *kd_tree_device.core.node_count;

#if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) && DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG
	unsigned int lightcut_count		= *kd_tree_device.learning_to_cluster.lightcut_count;
	unsigned int reset_thread_count = std::max(node_count, lightcut_count);
	for (unsigned int reset_index = 0; reset_index < reset_thread_count; reset_index++)
		IlluminationAwareKDTree_LearningToClusterResetBatchKDTreeAndLightcutStatistics(kd_tree_device, reset_index);

	if (m_render_data.render_settings.sample_number == 0)
	{
		const LightTreeSGBuildResult<std::vector>& light_tree_sg_build_result			 = m_light_tree_sg_build_result;
		const std::vector<unsigned int>& second_tree_cut_node_indices					 = light_tree_sg_build_result.second_tree_cut_node_indices;
		unsigned int effective_second_tree_cut_size										 = light_tree_sg_build_result.effective_second_tree_cut_size;
		kd_tree_device.learning_to_cluster.effective_initial_lightcut_size				 = effective_second_tree_cut_size;
		m_render_data.kd_tree_device.learning_to_cluster.effective_initial_lightcut_size = effective_second_tree_cut_size;

		std::fill(m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_learning_to_cluster_data.m_initial_lightcut_node_indices.begin(),
				  m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_learning_to_cluster_data.m_initial_lightcut_node_indices.end(),
				  IlluminationAwareKDTreeNode::INVALID_NODE_INDEX);
		std::copy_n(second_tree_cut_node_indices.begin(), effective_second_tree_cut_size,
					m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_learning_to_cluster_data.m_initial_lightcut_node_indices.begin());

		if (effective_second_tree_cut_size > 0)
		{
			LightTreeSGDevice light_tree_sg = m_render_data.light_tree_sg;
			for (unsigned int slot = 0; slot < LearningToClusterMaximumLightCutSize; slot++)
				IlluminationAwareKDTree_LearningToClusterInitializeRootLightcut(kd_tree_device, light_tree_sg, slot);
		}
	}
#else  // #if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) && DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG
	for (unsigned int reset_index = 0; reset_index < node_count; reset_index++)
		IlluminationAwareKDTree_CoreResetBatchKDTreeStatistics(kd_tree_device, reset_index);
#endif // #if DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) && DirectLightSamplingStrategy == LSS_BASE_LIGHT_TREE_SG
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)

	// Resetting the status buffers
	// Uploading false to reset the flag
	*m_render_data.aux_buffers.still_one_ray_active = false;
	// Resetting the counter of pixels converged to 0
	m_render_data.aux_buffers.pixel_count_converged_so_far->store(0);

	if (frame_number > m_render_data.nee_plus_plus.m_stop_update_samples)
		m_render_data.nee_plus_plus.m_update_visibility_map = false;
}

void CPURenderer::post_sample_update(int frame_number)
{
#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	if (m_render_data.nisml.learning_enabled && m_nisml_state.m_training_record_percentage > 0.0f)
	{
		unsigned int record_count = m_nisml_state.m_nisml_data.get_effective_training_record_count();
		if (record_count > 0u)
		{
			NeuralImportanceSamplingMLP mlp = m_nisml_state.m_mlp.to_device(m_nisml_state.m_adam_learning_rate);
			NISMLPositionLearnableDenseGridDevice position_learnable_dense_grid =
				m_nisml_state.m_position_learnable_dense_grid.to_device(m_nisml_state.m_adam_learning_rate);

			for (unsigned int record_index = 0; record_index < record_count; record_index++)
				NISMLTrain(mlp, m_render_data, nullptr, record_count, record_index);

			unsigned int training_sample_count =
				m_nisml_state.m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>()[0].load();
			if (training_sample_count > 0u)
			{
				for (unsigned int connection_index = 0; connection_index < NeuralImportanceSamplingMLP::CONNECTIONS_COUNT; connection_index++)
					NISMLOptimize(mlp, m_nisml_state.m_adam_step, connection_index);

				for (unsigned int feature_index = 0; feature_index < NISML_POSITION_LEARNABLE_DENSE_GRID_TOTAL_PARAMETER_COUNT; feature_index++)
					NISMLGridOptimize(position_learnable_dense_grid, training_sample_count, m_nisml_state.m_adam_step, feature_index);

				m_nisml_state.m_adam_step++;
			}

			m_nisml_state.m_mlp.m_mlp_data.template get_buffer<MLPDataHostBuffers::MLP_LAST_TRAINING_SAMPLE_COUNT>()[0].store(0u);
		}
	}
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS

	m_render_data.render_settings.need_to_reset = false;
	// We want the G Buffer of the frame that we just rendered to go in the "g_buffer_prev_frame"
	// and then we can re-use the old buffers of to be filled by the current frame render

	GMoN_post_sample_update();
	ReSTIR_PT_post_sample_update();
	illumination_aware_kd_tree_post_sample_update();

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_nisml_state.m_nisml_data.reset();
#endif

	if (m_render_data.render_settings.accumulate)
		m_render_data.render_settings.sample_number++;
}

void CPURenderer::update_cameras(int sample)
{
	m_render_data.prev_camera	 = m_render_data.current_camera;
	m_render_data.current_camera = m_camera.to_hiprt(m_resolution.x, m_resolution.y);
}

void CPURenderer::reset()
{
	m_render_data.render_settings.need_to_reset = true;
	m_render_data.render_settings.sample_number = 0;

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_nisml_state.m_nisml_data.reset();
	m_nisml_state.m_mlp.initialize(true);
	m_nisml_state.m_position_learnable_dense_grid.initialize();
	m_nisml_state.m_adam_step = 0;
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
}

void CPURenderer::illumination_aware_kd_tree_reset()
{
#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_nisml_data.clear_representative_metadata();
	m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.reset();
	m_illumination_aware_kd_tree_state.lookahead_frontier_initialized	  = false;
	m_illumination_aware_kd_tree_state.current_frontier_uses_first_buffer = true;
	m_illumination_aware_kd_tree_state.next_creation_tag				  = 0;

	m_render_data.kd_tree_device = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.to_device(m_render_data);

	for (unsigned int node_index = 0; node_index < m_render_data.kd_tree_device.core.node_capacity; node_index++)
		IlluminationAwareKDTree_CoreResetTree(m_render_data.kd_tree_device, m_scene_bounding_box.mini, m_scene_bounding_box.maxi, node_index);

#elif DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.reset();
	m_illumination_aware_kd_tree_state.lookahead_frontier_initialized	  = false;
	m_illumination_aware_kd_tree_state.current_frontier_uses_first_buffer = true;
	m_illumination_aware_kd_tree_state.next_creation_tag				  = 0;

	m_render_data.kd_tree_device = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.to_device(m_render_data);
	unsigned int reset_count =
		std::max(m_render_data.kd_tree_device.core.node_capacity,
				 static_cast<unsigned int>(m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_learning_to_cluster_data.m_lightcut_data.size()));
	for (unsigned int node_index = 0; node_index < reset_count; node_index++)
		IlluminationAwareKDTree_CoreResetTree(m_render_data.kd_tree_device, m_scene_bounding_box.mini, m_scene_bounding_box.maxi, node_index);

#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
}

void CPURenderer::illumination_aware_kd_tree_post_sample_update()
{
#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
	IlluminationAwareKDTreeDevice kd_tree_device = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.to_device(m_render_data);

	if (m_render_data.render_settings.sample_number <= kd_tree_device.core.user_settings.stop_refining_after_SPP - 1)
	{
		unsigned int sample_count = kd_tree_device.core.training_sample_count->load();

		for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
			IlluminationAwareKDTree_CoreAccumulateBatchTrainingSamples(kd_tree_device, sample_index);

		unsigned int node_count = *kd_tree_device.core.node_count;
		for (unsigned int reduction_level = IlluminationAwareKDTreeMaximumLookaheadLevelCount; reduction_level > 0; reduction_level--)
			for (unsigned int node_index = 0; node_index < node_count; node_index++)
				IlluminationAwareKDTree_CoreReduceBatchStatisticsUpward(kd_tree_device, reduction_level, node_index);

		for (unsigned int node_index = 0; node_index < node_count; node_index++)
			IlluminationAwareKDTree_CoreAccumulateBatchStatisticsIntoHistory(kd_tree_device, node_index);

		unsigned int* current_frontier					 = kd_tree_device.core.active_guiding_nodes;
		AtomicType<unsigned int>* current_frontier_count = kd_tree_device.core.active_guiding_node_count;
		unsigned int* next_frontier = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.data();
		AtomicType<unsigned int>* next_frontier_count =
			m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.data();
		bool next_frontier_uses_first_buffer = true;

		for (unsigned int depth = 0; depth < IlluminationAwareKDTreeMaximumLookaheadLevelCount; depth++)
		{
			kd_tree_device.core.current_frontier	   = current_frontier;
			kd_tree_device.core.current_frontier_count = current_frontier_count;
			kd_tree_device.core.next_frontier		   = next_frontier;
			kd_tree_device.core.next_frontier_count	   = next_frontier_count;

			AtomicType<unsigned int>* frontier_count_to_clear;
			if (depth + 1 == IlluminationAwareKDTreeMaximumLookaheadLevelCount)
				frontier_count_to_clear = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.data();
			else if (next_frontier_uses_first_buffer)
				frontier_count_to_clear = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.data();
			else
				frontier_count_to_clear = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.data();

			*kd_tree_device.core.node_count_before_expansion = IlluminationAwareKDTreeNode::INVALID_NODE_INDEX;

			const unsigned int creation_tag					= m_illumination_aware_kd_tree_state.next_creation_tag++;
			const unsigned int current_frontier_count_value = current_frontier_count->load();
			for (unsigned int frontier_index = 0; frontier_index < current_frontier_count_value; frontier_index++)
				IlluminationAwareKDTree_CoreExpandOneLookaheadLevel(kd_tree_device, creation_tag, frontier_index);

			for (unsigned int sample_index = 0; sample_index < sample_count; sample_index++)
				IlluminationAwareKDTree_CoreReplayTrainingSamples(kd_tree_device, creation_tag, sample_index);

			const unsigned int updated_node_count = *kd_tree_device.core.node_count;
			for (unsigned int node_index = 0; node_index < updated_node_count; node_index++)
				IlluminationAwareKDTree_CoreInitializeCreatedNodeHistory(kd_tree_device, creation_tag, frontier_count_to_clear, node_index);

			const unsigned int next_frontier_count_value = next_frontier_count->load();
			current_frontier							 = next_frontier;
			current_frontier_count						 = next_frontier_count;
			if (next_frontier_uses_first_buffer)
			{
				next_frontier		= m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier.data();
				next_frontier_count = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_next_frontier_count.data();
			}
			else
			{
				next_frontier		= m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier.data();
				next_frontier_count = m_illumination_aware_kd_tree_state.illumination_aware_kd_tree.m_kd_tree_data.m_current_frontier_count.data();
			}
			next_frontier_uses_first_buffer = !next_frontier_uses_first_buffer;

			if (next_frontier_count_value == 0)
				break;
		}

		const unsigned int active_guiding_node_count = kd_tree_device.core.active_guiding_node_count->load();
		for (unsigned int guiding_list_index = 0; guiding_list_index < active_guiding_node_count; guiding_list_index++)
			IlluminationAwareKDTree_CoreMarkGuidingCellsForSplitting(kd_tree_device, guiding_list_index);

		for (unsigned int guiding_list_index = 0; guiding_list_index < active_guiding_node_count; guiding_list_index++)
			IlluminationAwareKDTree_CorePromoteGuidingCells(kd_tree_device, active_guiding_node_count, guiding_list_index);
	}

#if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	unsigned int training_record_capacity = m_render_data.nisml.training_record_capacity;
	for (unsigned int record_index = 0; record_index < training_record_capacity; record_index++)
		IlluminationAwareKDTree_NISMLReplayTrainingSamples(m_render_data, record_index);

	if (kd_tree_device.nisml.nisml_pending_cell_count->load() > 0u)
		for (unsigned int cache_index = 0; cache_index < kd_tree_device.nisml.nisml_hash_table_capacity; cache_index++)
			IlluminationAwareKDTree_NISMLBuildCaches(kd_tree_device, m_render_data, cache_index);
#elif DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator) // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
	unsigned int lightcut_sample_count = kd_tree_device.learning_to_cluster.training_sample_count->load();
	for (unsigned int sample_index = 0; sample_index < lightcut_sample_count; sample_index++)
		IlluminationAwareKDTree_LearningToClusterAccumulateNormalFaceObservations(kd_tree_device, sample_index);

	unsigned int active_guiding_count = kd_tree_device.core.active_guiding_node_count->load();
	for (unsigned int active_guiding_node_face_index = 0; active_guiding_node_face_index < active_guiding_count * SurfaceNormalFace_Count;
		 active_guiding_node_face_index++)
		IlluminationAwareKDTree_LearningToClusterAllocateNormalFaceLightcuts(kd_tree_device, active_guiding_node_face_index);

	for (unsigned int sample_index = 0; sample_index < lightcut_sample_count; sample_index++)
		IlluminationAwareKDTree_LearningToClusterInitializeShadingContexts(kd_tree_device, sample_index);

	unsigned int lightcut_count		= kd_tree_device.learning_to_cluster.lightcut_count->load();
	LightTreeSGDevice light_tree_sg = m_render_data.light_tree_sg;
	for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterInitializeLightcutQ0(kd_tree_device, light_tree_sg, static_cast<int>(lightcut_index));
	unsigned int reset_sample_counts = 1u;
	for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterResetBatchLightcutStatistics(kd_tree_device, reset_sample_counts, static_cast<int>(lightcut_index));
	for (unsigned int sample_index = 0; sample_index < lightcut_sample_count; sample_index++)
		IlluminationAwareKDTree_LearningToClusterReplayStatistics(kd_tree_device, light_tree_sg, static_cast<int>(sample_index));
	IlluminationAwareKDTreeStatisticsUpdateMode statistics_update_mode = IlluminationAwareKDTreeStatisticsUpdateMode::UPDATE_ALL;
	for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterStatisticsUpdates(kd_tree_device, statistics_update_mode, static_cast<int>(lightcut_index));

	int lightcut_refinement_rounds_per_SPP = std::max(m_illumination_aware_kd_tree_state.lightcut_refinement_rounds_per_SPP, 1);
	for (int refinement_round = 0; refinement_round < lightcut_refinement_rounds_per_SPP; refinement_round++)
	{
		for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
			IlluminationAwareKDTree_LearningToClusterRefineLightcuts(kd_tree_device, light_tree_sg, static_cast<int>(lightcut_index));

		if (refinement_round + 1 >= lightcut_refinement_rounds_per_SPP)
			break;

		reset_sample_counts = 1u;
		for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
			IlluminationAwareKDTree_LearningToClusterResetBatchLightcutStatistics(kd_tree_device, reset_sample_counts, static_cast<int>(lightcut_index));
		for (unsigned int sample_index = 0; sample_index < lightcut_sample_count; sample_index++)
			IlluminationAwareKDTree_LearningToClusterReplayStatistics(kd_tree_device, light_tree_sg, static_cast<int>(sample_index));

		statistics_update_mode = IlluminationAwareKDTreeStatisticsUpdateMode::INITIALIZE_EMPTY_ONLY;
		for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
			IlluminationAwareKDTree_LearningToClusterStatisticsUpdates(kd_tree_device, statistics_update_mode, static_cast<int>(lightcut_index));
	}

	reset_sample_counts = 0u;
	for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterResetBatchLightcutStatistics(kd_tree_device, reset_sample_counts, static_cast<int>(lightcut_index));
	for (unsigned int sample_index = 0; sample_index < lightcut_sample_count; sample_index++)
		IlluminationAwareKDTree_LearningToClusterReplayQRewards(kd_tree_device, light_tree_sg, static_cast<int>(sample_index));
	for (unsigned int lightcut_index = 0; lightcut_index < lightcut_count; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterQUpdates(kd_tree_device, static_cast<int>(lightcut_index));
	for (unsigned int lightcut_index = 0; lightcut_index < kd_tree_device.learning_to_cluster.lightcut_capacity; lightcut_index++)
		IlluminationAwareKDTree_LearningToClusterBuildLightcutSamplingCDFs(kd_tree_device, light_tree_sg, static_cast<int>(lightcut_index));
#endif																   // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS
#endif // #if DirectLightNEEEstimator == LSS_NEURAL_MANY_LIGHTS || DIRECT_LIGHT_NEE_IS_LEARNING_TO_CLUSTER(DirectLightNEEEstimator)
}

void CPURenderer::debug_render_pass(std::function<void(int, int)> render_pass_function)
{
	// Center pixel when rendering a neighborhood
	int center_x = 0;
	int center_y = 0;

	// If we want to debug a pixel that is not the center pixel,
	// the coordinates will be stored there
	int debug_x = -1;
	int debug_y = -1;

#if DEBUG_PIXEL

#if DEBUG_FLIP_Y
	center_x = DEBUG_PIXEL_X;
	center_y = DEBUG_PIXEL_Y;

	debug_x = center_x;
	debug_y = center_y;
#else  // DEBUG_FLIP_Y // #if DEBUG_FLIP_Y
	center_x = DEBUG_PIXEL_X;
	center_y = m_resolution.y - DEBUG_PIXEL_Y - 1;

	debug_x = center_x;
	debug_y = center_y;
#endif // DEBUG_FLIP_Y // #if DEBUG_FLIP_Y

#if DEBUG_OTHER_PIXEL_X != -1 && DEBUG_OTHER_PIXEL_Y != -1
#if DEBUG_OTHER_FLIP_Y
	debug_x = DEBUG_OTHER_PIXEL_X;
	debug_y = DEBUG_OTHER_PIXEL_Y;
#else  // DEBUG_OTHER_FLIP_Y
	debug_x = DEBUG_OTHER_PIXEL_X;
	debug_y = m_resolution.y - DEBUG_OTHER_PIXEL_Y - 1;
#endif // DEBUG_OTHER_FLIP_Y // #if DEBUG_OTHER_FLIP_Y
#endif // DEBUG_OTHER_PIXEL_X != -1 && DEBUG_OTHER_PIXEL_Y != -1 // #if DEBUG_OTHER_PIXEL_X != -1 && DEBUG_OTHER_PIXEL_Y != -1

	// Debugging the chosen pixel first
	render_pass_function(debug_x, debug_y);

#if DEBUG_RENDER_NEIGHBORHOOD
	// Rendering the neighborhood

#pragma omp parallel for schedule(dynamic)
	for (int render_y = std::max(0, center_y - DEBUG_NEIGHBORHOOD_SIZE); render_y <= std::min(m_resolution.y - 1, center_y + DEBUG_NEIGHBORHOOD_SIZE);
		 render_y++)
	{
		for (int render_x = std::max(0, center_x - DEBUG_NEIGHBORHOOD_SIZE); render_x <= std::min(m_resolution.x - 1, center_x + DEBUG_NEIGHBORHOOD_SIZE);
			 render_x++)
		{
			if (render_x == debug_x && render_y == debug_y)
				// Skipping the pixel that we debugged to avoid rendering it twice
				continue;

			render_pass_function(render_x, render_y);
		}
	}
#endif // DEBUG_RENDER_NEIGHBORHOOD // #if DEBUG_RENDER_NEIGHBORHOOD

#else // DEBUG_PIXEL // #if DEBUG_PIXEL

#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < m_resolution.y; y++)
	{
		for (int x = 0; x < m_resolution.x; x++)
		{
			if (x == debug_x && y == debug_y)
				// Skipping the pixel that we debugged to avoid rendering it twice
				continue;

			render_pass_function(x, y);
		}
	}

#endif // DEBUG_PIXEL // #if DEBUG_PIXEL
}

void CPURenderer::nee_plus_plus_cache_visibility_pass()
{
#if DirectLightUseNEEPlusPlus == KERNEL_OPTION_FALSE
	return;
#endif

	debug_render_pass([this](int x, int y) { NEEPlusPlus_Grid_Prepopulate(m_render_data, x, y); });
}

void CPURenderer::camera_rays_pass()
{
	debug_render_pass([this](int x, int y) { CameraRays(m_render_data, x, y); });
}

void CPURenderer::hierarchical_adaptive_sampling_pass()
{
	HIPRTRenderSettings& render_settings = m_render_data.render_settings;
	if (!render_settings.use_hierarchical_adaptive_sampling())

		return;

	unsigned int completed_sample_count = render_settings.sample_number + 1u;
	unsigned int minimum_sample_count	= static_cast<unsigned int>(std::max(2, render_settings.adaptive_sampling_min_samples));
	// Rebuilding periodically allows completed regions to become active again when their error estimate changes. Rebuilding too often is wasteful, but
	// rebuilding too infrequently can cause the adaptive sampling to miss regions that have changed. A rebuild interval of 2% of the total sample count is a
	// reasonable compromise.
	unsigned int rebuild_interval = 0.02f * m_render_data.render_settings.sample_number;
	if ((rebuild_interval & 1u) != 0u)
		rebuild_interval++;

	if (completed_sample_count < minimum_sample_count || completed_sample_count % rebuild_interval != 0u)

		return;

	for (int y = 0; y < m_resolution.y; y++)
		for (int x = 0; x < m_resolution.x; x++)
			HierarchicalAdaptiveSamplingComputeError(m_render_data, x, y);

	for (int y = 0; y < m_resolution.y; y++)
		HierarchicalAdaptiveSamplingBuildRows(m_render_data, y);

	for (int x = 0; x < m_resolution.x; x++)
		HierarchicalAdaptiveSamplingBuildColumns(m_render_data, x);

	unsigned int level_node_count													  = 0;
	HIPRTRenderData hierarchy_render_data											  = m_render_data;
	hierarchy_render_data.aux_buffers.hierarchical_adaptive_sampling_level_node_count = &level_node_count;

	unsigned int maximum_depth = static_cast<unsigned int>(std::max(1, render_settings.hierarchical_adaptive_sampling_max_depth));
	HierarchicalAdaptiveSamplingBuildHierarchy(hierarchy_render_data, 0u, static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::INITIALIZE));

	for (unsigned int build_depth = 0; build_depth <= maximum_depth; build_depth++)
	{
		HierarchicalAdaptiveSamplingBuildHierarchy(hierarchy_render_data, 0u,
												   static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::PREPARE_LEVEL));

		unsigned int maximum_thread_count = static_cast<unsigned int>(m_hierarchical_adaptive_sampling_nodes.size());
		unsigned int thread_count		  = maximum_thread_count;
		if (build_depth < 30u)
		{
			unsigned int maximum_nodes_through_depth = (1u << (build_depth + 1u)) - 1u;
			thread_count							 = std::min(maximum_thread_count, maximum_nodes_through_depth);
		}

		for (unsigned int thread_index = 0; thread_index < thread_count; thread_index++)
			HierarchicalAdaptiveSamplingBuildHierarchy(hierarchy_render_data, thread_index, build_depth);
	}

	HierarchicalAdaptiveSamplingBuildHierarchy(hierarchy_render_data, 0u, static_cast<unsigned int>(HierarchicalAdaptiveSamplingBuildCommand::FINALIZE));

	for (int y = 0; y < m_resolution.y; y++)
		for (int x = 0; x < m_resolution.x; x++)
			HierarchicalAdaptiveSamplingResolveMask(m_render_data, x, y);
}

void CPURenderer::ReSTIR_GI_pass()
{
	compute_ReSTIR_GI_optimal_spatial_reuse_radii();

	configure_ReSTIR_GI_initial_candidates_pass();
	launch_ReSTIR_GI_initial_candidates_pass();

	configure_ReSTIR_GI_temporal_reuse_pass();
	launch_ReSTIR_GI_temporal_reuse_pass();

	for (int i = 0; i < m_render_data.render_settings.restir_gi_settings.common_spatial_pass.number_of_passes; i++)
	{
		configure_ReSTIR_GI_spatial_reuse_pass(i);
		launch_ReSTIR_GI_spatial_reuse_pass();
	}

	configure_ReSTIR_GI_shading_pass();
	launch_ReSTIR_GI_shading_pass();
}

void CPURenderer::ReSTIR_PT_pass()
{
	compute_ReSTIR_PT_optimal_spatial_reuse_radii();

	configure_ReSTIR_PT_initial_candidates_pass();
	launch_ReSTIR_PT_initial_candidates_pass();

	configure_ReSTIR_PT_temporal_reuse_pass();
	launch_ReSTIR_PT_temporal_reuse_pass();

	for (int i = 0; i < m_render_data.render_settings.restir_pt_settings.common_spatial_pass.number_of_passes; i++)
	{
		configure_ReSTIR_PT_spatial_reuse_pass(i);
		launch_ReSTIR_PT_spatial_reuse_pass();
	}

	configure_ReSTIR_PT_shading_pass();
	launch_ReSTIR_PT_shading_pass();
}

void CPURenderer::ReSTIR_PG_pass()
{
	// Splatting
#pragma omp parallel for
	for (int index = 0; index < m_render_data.render_settings.render_resolution.x * m_render_data.render_settings.render_resolution.y; index++)
	{
		ReSTIR_PG_Splatting(m_render_data, index);
	}

	// Fitting
	unsigned int grid_cell_alive_count = m_restir_pg_state.grid_cell_alive_count.load();
#pragma omp parallel for
	for (int index = 0; index < grid_cell_alive_count; index++)
	{
		ReSTIR_PG_Fitting(m_render_data, index);
	}

#pragma omp parallel for
	for (int index = 0; index < grid_cell_alive_count * ReSTIRPGDistributionComponentCount; index++)
	{
		ReSTIR_PG_ResetSufficientStatistics(m_render_data, index);
	}
}

void CPURenderer::tracing_pass()
{
	debug_render_pass([this](int x, int y) { MegaKernel(m_render_data, x, y); });
}

void CPURenderer::compute_ReSTIR_GI_optimal_spatial_reuse_radii()
{
	debug_render_pass(
		[this](int x, int y)
		{
			ReSTIR_Directional_Reuse_Compute<true>(
				m_render_data, x, y, m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull,
				m_render_data.render_settings.restir_gi_settings.common_spatial_pass.per_pixel_spatial_reuse_radius);
		});
}

void CPURenderer::configure_ReSTIR_GI_initial_candidates_pass()
{
	m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer = m_restir_gi_state.initial_candidates_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_GI_initial_candidates_pass()
{
	if (m_render_data.render_settings.nb_bounces > 0)
	{
		debug_render_pass([this](int x, int y) { ReSTIR_GI_InitialCandidates(m_render_data, x, y); });
	}
}

void CPURenderer::configure_ReSTIR_GI_temporal_reuse_pass()
{
	if (m_render_data.render_settings.sample_number == 0)
		// First frame, using the initial candidates as the input
		m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs =
			m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;
	else
		// Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
		m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs =
			m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs;

	// For the output, using whatever buffer isn't the one we're reading from (the input buffer)
	if (m_render_data.render_settings.restir_gi_settings.temporal_pass.input_reservoirs == m_restir_gi_state.temporal_reservoirs.data())
		m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_restir_gi_state.spatial_reservoirs.data();
	else
		m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs = m_restir_gi_state.temporal_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_GI_temporal_reuse_pass()
{
	if (m_render_data.render_settings.nb_bounces > 0 && m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
	{
		debug_render_pass([this](int x, int y) { ReSTIR_GI_TemporalReuse(m_render_data, x, y); });
	}
}

void CPURenderer::configure_ReSTIR_GI_spatial_reuse_pass(int spatial_pass_index)
{
	m_render_data.render_settings.restir_gi_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	// The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
	// stores in the 'spatial buffer'

	ReSTIRGIReservoir* input_reservoirs;
	ReSTIRGIReservoir* output_reservoirs;

	if (spatial_pass_index > 0)
		// If this is the second spatial reuse pass or more, reading from the output of the previous pass
		input_reservoirs = m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs;
	else
	{
		// This is the first spatial reuse pass, reading from the output of the temporal pass
		// or the initial candidates depending on whether or not we have a temporal reuse pass at all

		if (m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
			// and we have a temporal reuse pass so we're going to read from the temporal reservoirs
			input_reservoirs = m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
		else
			// and we do not have a temporal reuse pass so we're just going to read from the initial candidates
			input_reservoirs = m_restir_gi_state.initial_candidates_reservoirs.data();
	}

	// Outputting to whichever reservoir we're not reading from to avoid race conditions
	if (input_reservoirs == m_restir_gi_state.temporal_reservoirs.data())
		output_reservoirs = m_restir_gi_state.spatial_reservoirs.data();
	else
		output_reservoirs = m_restir_gi_state.temporal_reservoirs.data();

	m_render_data.render_settings.restir_gi_settings.spatial_pass.input_reservoirs	= input_reservoirs;
	m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs = output_reservoirs;
}

void CPURenderer::launch_ReSTIR_GI_spatial_reuse_pass()
{
	debug_render_pass([this](int x, int y) { ReSTIR_GI_SpatialReuse(m_render_data, x, y); });
}

void CPURenderer::configure_ReSTIR_GI_shading_pass()
{
	if (m_render_data.render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass)
		m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_gi_settings.spatial_pass.output_reservoirs;
	else if (m_render_data.render_settings.restir_gi_settings.common_temporal_pass.do_temporal_reuse_pass)
		m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_gi_settings.temporal_pass.output_reservoirs;
	else
		m_render_data.render_settings.restir_gi_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_gi_settings.initial_candidates.initial_candidates_buffer;
}

void CPURenderer::launch_ReSTIR_GI_shading_pass()
{
	debug_render_pass([this](int x, int y) { ReSTIR_GI_Shading(m_render_data, x, y); });
}

void CPURenderer::compute_ReSTIR_PT_optimal_spatial_reuse_radii()
{
	debug_render_pass(
		[this](int x, int y)
		{
			ReSTIR_Directional_Reuse_Compute<true>(
				m_render_data, x, y, m_render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_directions_mask_ull,
				m_render_data.render_settings.restir_pt_settings.common_spatial_pass.per_pixel_spatial_reuse_radius);
		});
}

void CPURenderer::configure_ReSTIR_PT_initial_candidates_pass()
{
	m_render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer = m_restir_pt_state.initial_candidates_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_PT_initial_candidates_pass()
{
	debug_render_pass([this](int x, int y) { ReSTIR_PT_InitialCandidates(m_render_data, x, y); });
}

void CPURenderer::launch_ReSTIR_PT_spmis_create_reuse_cells_pass(ReSTIRPTReservoir* input_reservoirs)
{
	ReSTIRSPMISDataHostInternal<std::vector>& spmis_data = m_restir_pt_state.spmis_data.m_spmis_data;

	AtomicType<unsigned short int>* cell_pixels_counters = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_COUNTERS>().data();
	AtomicType<unsigned short int>* cell_non_zero_reservoir_counters =
		spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_NON_ZERO_RESERVOIR_COUNTERS>().data();
	AtomicType<unsigned int>* cell_confidence_sums		 = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CONFIDENCE_SUMS>().data();
	AtomicType<unsigned int>* cell_global_offset_counter = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_GLOBAL_OFFSET_COUNTER>().data();
	unsigned int num_cells								 = m_resolution.x * m_resolution.y;
	for (int i = 0; i < num_cells; i++)
	{
		ReSTIR_SPMIS_ResetCellsData(m_render_data, num_cells, i);
	}

	unsigned int* all_pixels_hashes	   = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_ALL_PIXEL_HASHES>().data();
	unsigned int* pixels_index_in_cell = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_ALL_PIXEL_INDEX_IN_CELL>().data();
	for (int i = 0; i < num_cells; i++)
	{
		// First call to count important pixels first
		ReSTIR_SPMIS_CountCells(all_pixels_hashes, pixels_index_in_cell, cell_pixels_counters, cell_non_zero_reservoir_counters, cell_confidence_sums,
								input_reservoirs, num_cells, true, i);
	}

	for (int i = 0; i < num_cells; i++)
	{
		// Second call to count non-important pixels
		ReSTIR_SPMIS_CountCells(all_pixels_hashes, pixels_index_in_cell, cell_pixels_counters, cell_non_zero_reservoir_counters, cell_confidence_sums,
								input_reservoirs, num_cells, false, i);
	}

	unsigned int* cell_offsets			= spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_OFFSETS>().data();
	void* compute_offsets_launch_args[] = { &cell_pixels_counters, &cell_global_offset_counter, &cell_offsets, &num_cells };
	for (int i = 0; i < num_cells; i++)
	{
		ReSTIR_SPMIS_ComputeOffsets(cell_pixels_counters, cell_global_offset_counter, cell_offsets, num_cells, i);
	}

	unsigned int* pixel_indices_sorted = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_PIXEL_INDICES_SORTED>().data();
	for (int i = 0; i < num_cells; i++)
	{
		ReSTIR_SPMIS_Sort(all_pixels_hashes, pixels_index_in_cell, cell_offsets, pixel_indices_sorted, num_cells, i);
	}

	unsigned int* cell_alive_list = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_ALIVE_LIST>().data();
	unsigned int cell_alive_count = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_TOTAL_COUNT_COUNTER>().at(0);
	float* cell_cdfs			  = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDFS>().data();

	if (spmis_data.template get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().size() < cell_alive_count * ReSTIR_PT_SPMISCDFLUTSize)
	{
		spmis_data.template resize_one_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>(cell_alive_count * ReSTIR_PT_SPMISCDFLUTSize);
		m_render_data.render_settings.restir_pt_settings.spmis_settings.cell_cdf_luts =
			spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().data();
	}
	unsigned short int* cell_cdf_luts = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUTS>().data();

	unsigned int* cell_cdf_luts_offsets = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_CDF_LUT_OFFSETS>().data();
	ReSTIR_SPMIS_BuildCDFs(cell_non_zero_reservoir_counters, cell_offsets, cell_alive_list, cell_alive_count, pixel_indices_sorted, input_reservoirs, cell_cdfs,
						   cell_cdf_luts, cell_cdf_luts_offsets);

	if (m_render_data.render_settings.sample_number == 0)
	{
		float* cell_variance = spmis_data.get_buffer<ReSTIRSPMISDataHostBuffers::RESTIR_SPMIS_CELL_VARIANCE>().data();
		ReSTIR_SPMIS_ComputeCellsVariance(cell_pixels_counters, cell_offsets, cell_alive_list, cell_alive_count, pixel_indices_sorted, input_reservoirs,
										  cell_variance);
	}
}

void CPURenderer::configure_ReSTIR_PT_temporal_reuse_pass()
{
	if (m_render_data.render_settings.sample_number == 0)
		// First frame, using the initial candidates as the input
		m_render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs =
			m_render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;
	else
		// Not the first frame, the input to the temporal pass is the output of the last frame ReSTIR
		m_render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs =
			m_render_data.render_settings.restir_pt_settings.restir_output_reservoirs;

	// For the output, using whatever buffer isn't the one we're reading from (the input buffer)
	if (m_render_data.render_settings.restir_pt_settings.temporal_pass.input_reservoirs == m_restir_pt_state.temporal_reservoirs.data())
		m_render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs = m_restir_pt_state.spatial_reservoirs.data();
	else
		m_render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs = m_restir_pt_state.temporal_reservoirs.data();
}

void CPURenderer::launch_ReSTIR_PT_temporal_reuse_pass()
{
	if (m_render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
	{
		debug_render_pass([this](int x, int y) { ReSTIR_PT_TemporalReuse(m_render_data, x, y); });
	}
}

void CPURenderer::configure_ReSTIR_PT_spatial_reuse_pass(int spatial_pass_index)
{
	m_render_data.render_settings.restir_pt_settings.common_spatial_pass.spatial_pass_index = spatial_pass_index;

	// The spatial reuse pass spatially reuse on the output of the temporal pass in the 'temporal buffer' and
	// stores in the 'spatial buffer'

	ReSTIRPTReservoir* input_reservoirs;
	ReSTIRPTReservoir* output_reservoirs;

	if (spatial_pass_index > 0)
		// If this is the second spatial reuse pass or more, reading from the output of the previous pass
		input_reservoirs = m_render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else
	{
		// This is the first spatial reuse pass, reading from the output of the temporal pass
		// or the initial candidates depending on whether or not we have a temporal reuse pass at all

		if (m_render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
			// and we have a temporal reuse pass so we're going to read from the temporal reservoirs
			input_reservoirs = m_render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
		else
			// and we do not have a temporal reuse pass so we're just going to read from the initial candidates
			input_reservoirs = m_restir_pt_state.initial_candidates_reservoirs.data();
	}

	// Outputting to whichever reservoir we're not reading from to avoid race conditions
	if (input_reservoirs == m_restir_pt_state.temporal_reservoirs.data())
		output_reservoirs = m_restir_pt_state.spatial_reservoirs.data();
	else
		output_reservoirs = m_restir_pt_state.temporal_reservoirs.data();

	m_render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs	= input_reservoirs;
	m_render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs = output_reservoirs;
}

void CPURenderer::launch_ReSTIR_PT_spatial_reuse_pass()
{
	if (ReSTIR_PT_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS ||
		ReSTIR_PT_MISWeightsType == RESTIR_MIS_WEIGHTS_TYPE_STOCHASTIC_PAIRWISE_MIS_DEFENSIVE)
	{
		launch_ReSTIR_PT_spmis_create_reuse_cells_pass(m_render_data.render_settings.restir_pt_settings.spatial_pass.input_reservoirs);

		debug_render_pass([this](int x, int y) { ReSTIR_PT_SpatialReuseSPMIS(m_render_data, x, y); });
	}
	else
		debug_render_pass([this](int x, int y) { ReSTIR_PT_SpatialReuse(m_render_data, x, y); });
}

void CPURenderer::configure_ReSTIR_PT_shading_pass()
{
	if (m_render_data.render_settings.restir_pt_settings.common_spatial_pass.do_spatial_reuse_pass)
		m_render_data.render_settings.restir_pt_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_pt_settings.spatial_pass.output_reservoirs;
	else if (m_render_data.render_settings.restir_pt_settings.common_temporal_pass.do_temporal_reuse_pass)
		m_render_data.render_settings.restir_pt_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_pt_settings.temporal_pass.output_reservoirs;
	else
		m_render_data.render_settings.restir_pt_settings.restir_output_reservoirs =
			m_render_data.render_settings.restir_pt_settings.initial_candidates.initial_candidates_buffer;
}

void CPURenderer::launch_ReSTIR_PT_shading_pass()
{
	debug_render_pass([this](int x, int y) { ReSTIR_PT_Shading(m_render_data, x, y); });
}

void CPURenderer::ReSTIR_PG_reset_hash_grid()
{
	for (int index = 0; index < m_restir_pg_state.hash_grid_distributions_soa_buffer.get_total_element_count(); index++)
	{
		ReSTIR_PG_ResetHashGrid(m_render_data, index);
	}
}

void CPURenderer::ReSTIR_PG_reset_distributions()
{
#if ReSTIRPGEnable == KERNEL_OPTION_FALSE
	return;
#endif

	for (int index = 0; index < m_restir_pg_state.hash_grid_distributions_soa_buffer.get_total_element_count() * ReSTIRPGDistributionComponentCount; index++)
	{
		ReSTIR_PG_ResetDistributions(m_render_data, index);
	}
}

void CPURenderer::gmon_compute_median_of_means()
{
	debug_render_pass([this](int x, int y) { GMoNComputeMedianOfMeans(m_render_data, x, y); });
}

void CPURenderer::tonemap(float gamma, float exposure)
{
	ColorRGB32F* framebuffer_data = get_framebuffer().get_data_as_ColorRGB32F();

#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < m_resolution.y; y++)
	{
		for (int x = 0; x < m_resolution.x; x++)
		{
			int index = x + y * m_resolution.x;

			ColorRGB32F hdr_color = framebuffer_data[index];

			if (m_render_data.render_settings.accumulate)
				// Scaling by sample count
				hdr_color = hdr_color / float(m_render_data.render_settings.sample_number);

			ColorRGB32F tone_mapped = ColorRGB32F(1.0f) - exp(-hdr_color * exposure);
			tone_mapped				= pow(tone_mapped, 1.0f / gamma);

			framebuffer_data[index] = tone_mapped;
		}
	}
}
