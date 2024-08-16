/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "HostDeviceCommon/RenderData.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Scene/SceneParser.h"
#include "Utils/CommandlineArguments.h"

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
    void camera_rays_pass();
    void ReSTIR_DI_initial_candidates_pass();
    void ReSTIR_DI_spatial_reuse_pass();
    void ReSTIR_DI_spatial_reuse_pass_internal();
    void tracing_pass();
    void tonemap(float gamma, float exposure);

private:
    int2 m_resolution;

    Image32Bit m_framebuffer;
    std::vector<unsigned char> m_pixel_active_buffer;
    std::vector<ColorRGB32F> m_denoiser_albedo;
    std::vector<float3> m_denoiser_normals;

    std::vector<int> m_pixel_sample_count;
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

    // Random number generator for given a random seed to the threads at each sample
    Xorshift32Generator m_rng;

    std::vector<Reservoir> m_restir_initial_reservoirs;
    std::vector<Reservoir> m_restir_temporal_reservoirs;
    std::vector<Reservoir> m_restir_final_reservoirs;

    std::vector<Triangle> m_triangle_buffer;
    std::shared_ptr<BVH> m_bvh;

    HIPRTCamera m_hiprt_camera;
    HIPRTRenderData m_render_data;
};

#endif
