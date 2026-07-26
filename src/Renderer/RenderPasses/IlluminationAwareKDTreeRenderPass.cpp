/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME = "Illumination-Aware KD-Tree Render Pass";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
}

void IlluminationAwareKDTreeRenderPass::resize(unsigned int new_width, unsigned int new_height) {}

bool IlluminationAwareKDTreeRenderPass::pre_render_update(float delta_time)
{
	if (m_illumination_aware_kd_tree.maximum_size() == 0)
	{
		m_illumination_aware_kd_tree.resize(1);

		return true;
	}

	return false;
}

bool IlluminationAwareKDTreeRenderPass::launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options)
{
	return false;
}

void IlluminationAwareKDTreeRenderPass::post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) {}

void IlluminationAwareKDTreeRenderPass::update_render_data()
{
	IlluminationAwareKDTreeDevice device = m_illumination_aware_kd_tree.to_device();

	device.subdivision_mode									 = m_renderer->get_render_data().illumination_aware_kd_tree.subdivision_mode;
	device.debug_counters									 = m_renderer->get_render_data().illumination_aware_kd_tree.debug_counters;
	m_renderer->get_render_data().illumination_aware_kd_tree = device;
}

void IlluminationAwareKDTreeRenderPass::reset(bool reset_by_camera_movement)
{
	m_illumination_aware_kd_tree.reset();
	m_renderer->get_render_data().illumination_aware_kd_tree.debug_counters = {};
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return true;
}
