/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/NISMLRenderPass.h"

#include "HostDeviceCommon/KernelOptions/DirectLightSamplingOptions.h"

const std::string NISMLRenderPass::NISML_RENDER_PASS_NAME = "Neural Importance Sampling MLP";

NISMLRenderPass::NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(NISMLRenderPass::NISML_RENDER_PASS_NAME, renderer, options)
{
}

void NISMLRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool NISMLRenderPass::pre_sample_update(float delta_time)
{
	if (!is_render_pass_used(*m_compiler_options))
		return false;

	if (m_mlp.maximum_size() == 0)
	{
		m_mlp.resize();
		m_mlp.initialize(false);
	}

	update_render_data();

	return false;
}

bool NISMLRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void NISMLRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void NISMLRenderPass::update_render_data()
{
	if (!is_render_pass_used(*m_compiler_options))
		return;

	HIPRTRenderData& render_data = m_renderer->get_render_data();
	render_data.nis_ml.mlp		 = m_mlp.to_device();
	render_data.scene_min		 = m_renderer->get_scene_metadata().scene_bounding_box.mini;
	render_data.scene_max		 = m_renderer->get_scene_metadata().scene_bounding_box.maxi;
}

void NISMLRenderPass::reset(bool reset_by_camera_movement) {}

bool NISMLRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return compiler_options.get_macro_value(GPUKernelCompilerOptions::DIRECT_LIGHT_NEE_ESTIMATOR) == LSS_NEURAL_MANY_LIGHTS;
}
