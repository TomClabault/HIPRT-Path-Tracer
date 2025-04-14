/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "Device/includes/ReSTIR/GI/Reservoir.h"
#include "Device/includes/ReSTIR/ReGIR/Settings.h"
#include "Device/kernel_parameters/ReSTIR/DI/LightPresamplingParameters.h"

#include "HostDeviceCommon/RenderData.h"

#include "Image/Image.h"
#include "Image/EnvmapRGBE9995.h"
#include "Renderer/BVH.h"
#include "Renderer/CPUDataStructures/GBufferCPUData.h"
#include "Renderer/CPUDataStructures/GMoNCPUData.h"
#include "Renderer/CPUDataStructures/NEEPlusPlusCPUData.h"
#include "Renderer/CPUDataStructures/MaterialPackedSoACPUData.h"
#include "Scene/SceneParser.h"
#include "Utils/CommandlineArguments.h"

#include <functional>
#include <memory>
#include <vector>

class CPURenderer
{
public:
    CPURenderer(int width, int height);

    void setup_brdfs_data();
    void setup_nee_plus_plus();
    void setup_gmon();
    void nee_plus_plus_memcpy_accumulation(int frame_number);
    void gmon_check_for_sets_accumulation();

    void set_scene(Scene& parsed_scene);
    void compute_emissives_power_area_alias_table(const Scene& scene);
    void set_envmap(Image32Bit& envmap_image);
    void set_camera(Camera& camera);

    HIPRTRenderData& get_render_data();
    HIPRTRenderSettings& get_render_settings();
    Image32Bit& get_framebuffer();

    void render();
    void pre_render_update(int frame_number);
    void update_render_data(int sample);

    void reset();

    void debug_render_pass(std::function<void(int, int)> render_pass_function);

    void nee_plus_plus_cache_visibility_pass();
    void camera_rays_pass();
    void ReGIR_grid_fill_pass();
    void ReSTIR_DI_pass();
    void ReSTIR_GI_pass();

    LightPresamplingParameters configure_ReSTIR_DI_light_presampling_pass();
    void configure_ReSTIR_DI_initial_pass();

    void launch_ReSTIR_DI_presampling_lights_pass();
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

    std::vector<AtomicType<float>> m_DEBUG_SUMS;
    std::vector<AtomicType<unsigned long long int>> m_DEBUG_SUM_COUNT;

private:
    int2 m_resolution;

    Image32Bit m_framebuffer;
    std::vector<unsigned char> m_pixel_active_buffer;
    std::vector<ColorRGB32F> m_denoiser_albedo;
    std::vector<float3> m_denoiser_normals;

    std::vector<int> m_pixel_sample_count;
    std::vector<int> m_pixel_converged_sample_count;
    std::vector<float> m_pixel_squared_luminance;
    unsigned char m_still_one_ray_active = true;
    AtomicType<unsigned int> m_stop_noise_threshold_count;

    RGBE9995Envmap<false> m_packed_envmap;
    std::vector<float> m_envmap_cdf;
    std::vector<float> m_envmap_alias_table_probas;
    std::vector<int> m_envmap_alias_table_alias;

    std::vector<float> m_power_area_alias_table_probas;
    std::vector<int> m_power_area_alias_table_alias;

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
        std::vector<ReSTIRDIPresampledLight> presampled_lights_buffer;

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
        ReGIRSettings settings;
        ReGIRGrid grid;

        std::vector<ReGIRReservoir> grid_buffer;
    } m_regir_state;

    Image32Bit m_sheen_ltc_params;
    Image32Bit m_GGX_conductor_directional_albedo;
    Image32Bit3D m_glossy_dielectrics_directional_albedo;
    Image32Bit3D m_GGX_glass_directional_albedo;
    Image32Bit3D m_GGX_glass_inverse_directional_albedo;
    Image32Bit3D m_GGX_thin_glass_directional_albedo;

    std::vector<Triangle> m_triangle_buffer;
    std::shared_ptr<BVH> m_bvh;

    Camera m_camera;
    HIPRTRenderData m_render_data;
};

#endif
