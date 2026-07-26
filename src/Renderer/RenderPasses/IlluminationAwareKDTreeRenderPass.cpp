/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/GPURenderer.h"
#include "Renderer/RenderPasses/IlluminationAwareKDTreeRenderPass.h"

#include "HostDeviceCommon/RenderData.h"

const std::string IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME = "Illumination-Aware KD-Tree Render Pass";
const std::string IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID				 = "Initialize Root Node";

IlluminationAwareKDTreeRenderPass::IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options)
	: RenderPass(IlluminationAwareKDTreeRenderPass::ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME, renderer, options)
{
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID] =
		std::make_shared<GPUKernel>(this->get_name() + "::" + IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID);
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->set_kernel_file_path(DEVICE_KERNELS_DIRECTORY
																									   "/IlluminationAwareKDTree/InitializeRootNode.h");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->set_kernel_function_name("initialize_illumination_tree_root");
	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->synchronize_options_with(m_compiler_options, {});
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
	HIPRTRenderData& render_data = m_renderer->get_render_data();

	render_data.illumination_aware_kd_tree					= m_illumination_aware_kd_tree.to_device();
	render_data.illumination_aware_kd_tree.subdivision_mode = m_subdivision_mode;
}

void IlluminationAwareKDTreeRenderPass::reset(bool reset_by_camera_movement)
{
	if (m_illumination_aware_kd_tree.maximum_size() == 0)
		// Nothing to reset
		return;

	m_illumination_aware_kd_tree.reset();

	HIPRTRenderData& render_data						  = m_renderer->get_render_data();
	render_data.illumination_aware_kd_tree.debug_counters = {};

	IlluminationAwareKDTreeNode* nodes		  = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODES>().data();
	IlluminationAwareKDTreeNodeBounds* bounds = m_illumination_aware_kd_tree.m_nodes_and_bounds.get_buffer<ILLUMINATION_AWARE_KD_TREE_NODE_BOUNDS>().data();
	uint32_t* node_count					  = m_illumination_aware_kd_tree.m_node_count.get_device_pointer();
	uint32_t* active_guiding_nodes			  = m_illumination_aware_kd_tree.m_active_guiding_nodes.get_device_pointer();
	uint32_t* active_guiding_node_count		  = m_illumination_aware_kd_tree.m_active_guiding_node_count.get_device_pointer();
	float3_t scene_bounds_minimum			  = m_renderer->get_scene_metadata().scene_bounding_box.mini;
	float3_t scene_bounds_maximum			  = m_renderer->get_scene_metadata().scene_bounding_box.maxi;
	void* launch_args[] = { &nodes, &bounds, &node_count, &active_guiding_nodes, &active_guiding_node_count, &scene_bounds_minimum, &scene_bounds_maximum };

	m_kernels[IlluminationAwareKDTreeRenderPass::INITIALIZE_ROOT_NODE_KERNEL_ID]->launch_asynchronous(1, 1, 1, 1, launch_args, m_renderer->get_main_stream());
}

bool IlluminationAwareKDTreeRenderPass::is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const
{
	return true;
}
