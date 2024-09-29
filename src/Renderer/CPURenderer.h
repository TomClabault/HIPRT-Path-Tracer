/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "Device/kernel_parameters/ReSTIR/DI/LightPresamplingParameters.h"
#include "HostDeviceCommon/RenderData.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Scene/SceneParser.h"
#include "Utils/CommandlineArguments.h"

#include <functional>
#include <memory>
#include <vector>

class CPURenderer
{
public:
    CPURenderer(int width, int height);

    void set_scene(Scene& parsed_scene);
    void set_envmap(Image32Bit& envmap_image);
    void set_camera(Camera& camera);

    HIPRTRenderData& get_render_data();
    HIPRTRenderSettings& get_render_settings();
    Image32Bit& get_framebuffer();

    void render();
    void update(int frame_number);
    void update_render_data(int sample);

    void debug_render_pass(std::function<void(int, int)> render_pass_function);
    void camera_rays_pass();

    void ReSTIR_DI();

    LightPresamplingParameters configure_ReSTIR_DI_light_presampling_pass();
    void configure_ReSTIR_DI_initial_pass();

    void launch_ReSTIR_DI_presampling_lights_pass();
    void launch_ReSTIR_DI_initial_candidates_pass();

    void configure_ReSTIR_DI_temporal_pass();
    void configure_ReSTIR_DI_temporal_pass_for_fused_spatiotemporal();
    void configure_ReSTIR_DI_spatial_pass(int spatial_pass_index);
    void configure_ReSTIR_DI_spatial_pass_for_fused_spatiotemporal(int spatial_pass_index);
    void configure_ReSTIR_DI_spatiotemporal_pass();
	void configure_ReSTIR_DI_output_buffer();

    void ReSTIR_DI_temporal_reuse_pass();
    void ReSTIR_DI_spatial_reuse_pass();
    void ReSTIR_DI_spatiotemporal_reuse_pass();

    void tracing_pass();

    void tonemap(float gamma, float exposure);

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

    struct GBuffer
    {
        std::vector<SimplifiedRendererMaterial> materials;
        std::vector<float3> geometric_normals;
        std::vector<float3> shading_normals;
        std::vector<float3> view_directions;
        std::vector<float3> first_hits;

        std::vector<unsigned char> cameray_ray_hit;

        std::vector<RayVolumeState> ray_volume_states;
    };

    GBuffer m_g_buffer;
    GBuffer m_g_buffer_prev_frame;

    // Random number generator for given a random seed to the threads at each sample
    Xorshift32Generator m_rng;

    struct ReSTIRDIState
    {
        std::vector<ReSTIRDIReservoir> initial_candidates_reservoirs;
        std::vector<ReSTIRDIReservoir> spatial_output_reservoirs_1;
        std::vector<ReSTIRDIReservoir> spatial_output_reservoirs_2;
        std::vector<ReSTIRDIPresampledLight> presampled_lights_buffer;

        ReSTIRDIReservoir* output_reservoirs = nullptr;


        bool odd_frame = false;
    } m_restir_di_state;

    std::vector<Triangle> m_triangle_buffer;
    std::shared_ptr<BVH> m_bvh;

    Camera m_camera;
    HIPRTRenderData m_render_data;
};

#endif
