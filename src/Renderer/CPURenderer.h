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
    void tonemap(float gamma, float exposure);
private:
    int2 m_resolution;

    Image32Bit m_framebuffer;
    std::vector<int> m_debug_pixel_active_buffer;
    std::vector<ColorRGB32F> m_denoiser_albedo;
    std::vector<float3> m_denoiser_normals;
    std::vector<int> m_pixel_sample_count;
    std::vector<float> m_pixel_squared_luminance;
    unsigned char m_still_one_ray_active = true;
    AtomicType<unsigned int> m_stop_noise_threshold_count;

    std::vector<Triangle> m_triangle_buffer;
    std::shared_ptr<BVH> m_bvh;

    HIPRTCamera m_hiprt_camera;
    HIPRTRenderData m_render_data;
};

#endif
