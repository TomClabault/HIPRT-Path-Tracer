/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"

#include "HostDeviceCommon/RenderData.h"

#include "Image/EnvmapRGBE9995.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Renderer/CPUDataStructures/GBufferCPUData.h"
#include "Renderer/CPUDataStructures/GMoNCPUData.h"
#include "Renderer/CPUDataStructures/MaterialPackedSoACPUData.h"
#include "Renderer/CPUDataStructures/NEEPlusPlusCPUData.h"
#include "Renderer/CPUGPUCommonDataStructures/BSDFDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/EmissiveMeshesAliasTablesHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRCellsLightDistributionsSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashCellDataSoAHost.h"
#include "Renderer/CPUGPUCommonDataStructures/ReGIRHashGridSoAHost.h"
#include "Renderer/LightTree/LightTreeATSBuilder.h"
#include "Renderer/LightTree/LightTreeSGBuilder.h"
#include "Scene/SceneParser.h"

#include <functional>
#include <memory>
#include <vector>

class CPURenderer
{
public:
	CPURenderer(int width, int height);

	void setup_bsdfs_data();

	void setup_nee_plus_plus();
	void setup_gmon();
	void GMoN_post_sample_update();
	void ReGIR_post_sample_update();

	void set_scene(Scene& parsed_scene);
	void compute_emissives_power_alias_table(const Scene& scene);
	void set_envmap(Image32Bit& envmap_image);
	void set_camera(Camera& camera);

	void resize_buffers();
	void update_render_data();
	void bsdfs_data_to_device();

	HIPRTRenderData& get_render_data();
	HIPRTRenderSettings& get_render_settings();
	Image32Bit& get_framebuffer();

	void render();
	void pre_render_update(int frame_number);
	void post_sample_update(int frame_number);
	void update_cameras(int sample);

	void reset();

	void debug_render_pass(std::function<void(int, int)> render_pass_function);

	void nee_plus_plus_cache_visibility_pass();
	void camera_rays_pass();
	void ReGIR_pass();
	void ReSTIR_DI_pass();
	void ReSTIR_GI_pass();

	template <bool accumulatePreIntegration>
	void ReGIR_grid_fill_pass(bool primary_hit);
	template <bool accumulatePreIntegration>
	ReGIRHashGridSoADevice ReGIR_spatial_reuse_pass(bool primary_hit);
	void ReGIR_pre_population();
	void ReGIR_pre_integration();
	void ReGIR_compute_cells_light_distributions();
	void ReGIR_compute_cells_light_distributions_internal(bool primary_hit);
	void ReGIR_compute_cell_light_compute_and_sort_internal(bool primary_hit, bool only_compute_sizes);

	void configure_ReSTIR_DI_initial_pass();
	void launch_ReSTIR_DI_initial_candidates_pass();

	void compute_ReSTIR_DI_optimal_spatial_reuse_radii();
	void configure_ReSTIR_DI_temporal_pass();
	void configure_ReSTIR_DI_temporal_pass_for_fused_spatiotemporal();
	void configure_ReSTIR_DI_spatial_pass(int spatial_pass_index);
	void configure_ReSTIR_DI_spatial_pass_for_fused_spatiotemporal(int spatial_pass_index);
	void configure_ReSTIR_DI_spatiotemporal_pass();
	void configure_ReSTIR_DI_output_buffer();

	void launch_ReSTIR_DI_temporal_reuse_pass();
	void launch_ReSTIR_DI_spatial_reuse_pass(int spatial_reuse_pass_index);
	void launch_ReSTIR_DI_spatiotemporal_reuse_pass();

	void tracing_pass();

	void compute_ReSTIR_GI_optimal_spatial_reuse_radii();
	void configure_ReSTIR_GI_initial_candidates_pass();
	void configure_ReSTIR_GI_temporal_reuse_pass();
	void configure_ReSTIR_GI_spatial_reuse_pass(int spatial_reuse_pass_index);
	void configure_ReSTIR_GI_shading_pass();

	void launch_ReSTIR_GI_initial_candidates_pass();
	void launch_ReSTIR_GI_temporal_reuse_pass();
	void launch_ReSTIR_GI_spatial_reuse_pass();
	void launch_ReSTIR_GI_shading_pass();

	void gmon_compute_median_of_means();

	void tonemap(float gamma, float exposure);

private:
	int2_t m_resolution;

	Image32Bit m_framebuffer;
	std::vector<ColorRGB32F> m_last_frame_ray_colors;

	std::vector<unsigned int> m_updated_random_seeds;
	std::vector<unsigned int> m_input_random_seeds;

	std::vector<unsigned char> m_pixel_active_buffer;
	std::vector<ColorRGB32F> m_denoiser_albedo;
	std::vector<float3_t> m_denoiser_normals;

	std::vector<int> m_pixel_sample_count;
	std::vector<int> m_pixel_converged_sample_count;
	std::vector<float> m_pixel_squared_luminance;
	unsigned char m_still_one_ray_active = true;
	AtomicType<unsigned int> m_stop_noise_threshold_count;

	RGBE9995Envmap<false> m_packed_envmap;
	std::vector<float> m_envmap_cdf;
	std::vector<float> m_envmap_alias_table_probas;
	std::vector<int> m_envmap_alias_table_alias;

	// Alias table for sampling emissive triangles in the scene proportional to
	// their power
	std::vector<float> m_power_alias_table_probas;
	std::vector<int> m_power_alias_table_alias;

	// Structure that contains an alias table for sampling an emissive mesh proportional
	// to its power as well as individual alias tables for each emissive mesh to be able
	// to sample an emissive triangle proportional to its power within a given mesh
	EmissiveMeshesAliasTablesHost<std::vector> m_emissive_meshes_alias_tables;

	LightTreeATSBuilder m_light_tree_builder_ats;
	LightTreeATSBuilderDeviceData<std::vector> m_light_tree_ats_device_data;

	LightTreeSGBuilder m_light_tree_builder_sg;
	LightTreeSGBuilderDeviceData<std::vector> m_light_tree_sg_device_data;

	NEEPlusPlusCPUData m_nee_plus_plus;

	GMoNCPUData m_gmon;

	DevicePackedTexturedMaterialSoACPUData m_gpu_packed_materials;
	// Keeps track of which material is fully opaque or not
	std::vector<unsigned char> m_material_opaque;

	GBufferCPUData m_g_buffer;
	GBufferCPUData m_g_buffer_prev_frame;

	// Random number generator for given a random seed to the threads at each sample
	Xorshift32Generator m_rng;

	struct ReSTIRDIState
	{
		std::vector<ReSTIRDIReservoir> initial_candidates_reservoirs;
		std::vector<ReSTIRDIReservoir> spatial_output_reservoirs_1;
		std::vector<ReSTIRDIReservoir> spatial_output_reservoirs_2;

		std::vector<unsigned int> per_pixel_spatial_reuse_directions_mask_u;
		std::vector<unsigned long long int> per_pixel_spatial_reuse_directions_mask_ull;
		std::vector<unsigned char> per_pixel_spatial_reuse_radius;

		AtomicType<unsigned long long int> spatial_reuse_hit_rate_hits;
		AtomicType<unsigned long long int> spatial_reuse_hit_rate_total;

		ReSTIRDIReservoir* output_reservoirs = nullptr;

		bool odd_frame = false;
	} m_restir_di_state;

	struct ReSTIRGIState
	{
		std::vector<ReSTIRGIReservoir> initial_candidates_reservoirs;
		std::vector<ReSTIRGIReservoir> temporal_reservoirs;
		std::vector<ReSTIRGIReservoir> spatial_reservoirs;

		std::vector<unsigned int> per_pixel_spatial_reuse_directions_mask_u;
		std::vector<unsigned long long int> per_pixel_spatial_reuse_directions_mask_ull;
		std::vector<unsigned char> per_pixel_spatial_reuse_radius;

		AtomicType<unsigned long long int> spatial_reuse_hit_rate_hits;
		AtomicType<unsigned long long int> spatial_reuse_hit_rate_total;
	} m_restir_gi_state;

	struct ReGIRState
	{
		ReGIRHashGridSoAHost<std::vector> grid_buffer_primary_hit;
		ReGIRHashGridSoAHost<std::vector> spatial_grid_buffer_primary_hit;
		ReGIRHashCellDataSoAHost<std::vector> hash_cell_data_primary_hit;

		ReGIRHashGridSoAHost<std::vector> grid_buffer_secondary_hit;
		ReGIRHashGridSoAHost<std::vector> spatial_grid_buffer_secondary_hit;
		ReGIRHashCellDataSoAHost<std::vector> hash_cell_data_secondary_hit;

		ReGIRHashGridSoAHost<std::vector> correlation_reduction_grid;

		std::vector<AtomicType<float>> non_canonical_pre_integration_factors_primary_hit;
		std::vector<AtomicType<float>> canonical_pre_integration_factors_primary_hit;

		std::vector<AtomicType<float>> non_canonical_pre_integration_factors_secondary_hit;
		std::vector<AtomicType<float>> canonical_pre_integration_factors_secondary_hit;

		ReGIRCellsLightDistributionsSoAHost<std::vector> cells_light_distributions_primary_hit;
		ReGIRCellsLightDistributionsSoAHost<std::vector> cells_light_distributions_secondary_hit;
		unsigned int m_last_cells_light_distributions_compute_count_primary_hits   = 0;
		unsigned int m_last_cells_light_distributions_compute_count_secondary_hits = 0;
		// Percentage of the total incoming energy that we should keep in each light distribution of
		// each cell at *most* (roughly)
		//
		// The light distribution will only contain as many emissive meshes as necessary such that the
		// distribution covers covers that percentage of the total incoming energy to the grid cell.
		//
		// This is "rounded up" so if 40% of the total incoming radiance is required by this parameter but
		// we have to choose between (for example):
		//
		// - 5 meshes in the distribution = 38% of the energy covered
		// - 6 meshes in the distribution = 51% of the energy covered
		//
		// Then the light distribution will cover 6 meshes
		float m_light_distribution_incoming_light_energy_target = 1.0f;

		std::vector<AtomicType<unsigned int>> grid_cell_alive;
		std::vector<unsigned int> grid_cells_alive_list;
		AtomicType<unsigned int> grid_cells_alive_count;
	} m_regir_state;

	BSDFDataHost m_bsdf_data_cpu_data;

	std::vector<Triangle> m_triangle_buffer;
	std::vector<Triangle> m_emissive_triangles_buffer;
	std::shared_ptr<BVH> m_bvh;
	// The light BVH is only used for tracing rays. This is a BVH built only over the emissive
	// triangles of the scene.
	//
	// This is not a light hierarchy for light sampling
	std::shared_ptr<BVH> m_light_bvh;

	Camera m_camera;
	HIPRTRenderData m_render_data;
};

#endif
