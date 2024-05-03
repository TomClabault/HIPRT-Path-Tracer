#ifndef CPU_RENDERER_H
#define CPU_RENDERER_H

#include "HostDeviceCommon/RenderData.h"
#include "Image/Image.h"
#include "Renderer/BVH.h"
#include "Scene/SceneParser.h"
#include "Utils/CommandlineArguments.h"

#include <vector>

class CPURenderer
{
public:
    CPURenderer(int width, int height);

    void set_scene(Scene& parsed_scene);
    void set_camera(Camera& camera);

    HIPRTRenderSettings& get_render_settings();
    Image& get_framebuffer();

    void render();
    void tonemap(float gamma, float exposure);
private:
    int2 m_resolution;

    Image m_framebuffer;
    std::vector<int> m_debug_pixel_active_buffer;
    std::vector<ColorRGB> m_denoiser_albedo;
    std::vector<float3> m_denoiser_normals;
    std::vector<int> m_pixel_sample_count;
    std::vector<float> m_pixel_squared_luminance;

    std::vector<Triangle> m_triangle_buffer;
    std::shared_ptr<BVH> m_bvh;

    HIPRTCamera m_hiprt_camera;
    HIPRTRenderData m_render_data;
};

#endif
