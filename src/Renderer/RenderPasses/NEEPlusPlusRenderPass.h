/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

 #ifndef NEE_PLUS_PLUS_RENDER_PASS_H
 #define NEE_PLUS_PLUS_RENDER_PASS_H
 
 #include "Renderer/GPUDataStructures/NEEPlusPlusGPUData.h"
 #include "Renderer/RenderPasses/RenderPass.h"
 
class NEEPlusPlusRenderPass : public RenderPass
{
public:
    static const std::string NEE_PLUS_PLUS_RENDER_PASS_NAME;
    static const std::string FINALIZE_ACCUMULATION_KERNEL_ID;
 
    NEEPlusPlusRenderPass();
    NEEPlusPlusRenderPass(GPURenderer* renderer);
    NEEPlusPlusRenderPass(GPURenderer* renderer, const std::string& name);
 
    virtual void resize(unsigned int new_width, unsigned int new_height) override {};
     
    virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx, const std::vector<hiprtFuncNameSet>& func_name_sets = {}, bool silent = false, bool use_cache = true) override;
    virtual bool pre_render_update(float delta_time) override;
    virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
    virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

    virtual void update_render_data() override;
    virtual void reset(bool reset_by_camera_movement);
 
    virtual bool is_render_pass_used() const override;

    NEEPlusPlusGPUData& get_nee_plus_plus_data();
 
private:
    // Buffers and settings for NEE++
	NEEPlusPlusGPUData m_nee_plus_plus;
};
 
#endif
 