/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
#define RENDERER_WAVEFRONT_RENDER_PASS_H

#include "HIPRT-Orochi/OrochiBuffer.h"
#include "Renderer/CPUGPUCommonDataStructures/WavefrontDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstddef>
#include <map>
#include <memory>
#include <string>

class WavefrontRenderPass : public RenderPass
{
public:
	static const std::string WAVEFRONT_RENDER_PASS_NAME;
	static const std::string SHADE_PRIMARY_PATHS_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_DIFFUSE_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_GLASS_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string SHADE_PRIMARY_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string SHADE_PATHS_KERNEL;
	static const std::string SHADE_PATHS_DIFFUSE_KERNEL;
	static const std::string SHADE_PATHS_GLASS_KERNEL;
	static const std::string SHADE_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string SHADE_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_DIFFUSE_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_GLASS_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_SINGLE_METALLIC_KERNEL;
	static const std::string COMPLETE_DEFERRED_PATHS_SPECULAR_DIFFUSE_KERNEL;
	static const std::string TRACE_PATHS_KERNEL;
	static const std::string TRACE_SHADOW_RAYS_KERNEL;
	static const std::string NEE_DEFERRED_MIS_CONTEXT_SIZE_KERNEL;
	static const std::string MATERIAL_FAMILY_ROUTING_CLASSIFY_SECONDARY_KERNEL;

	WavefrontRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);
	~WavefrontRenderPass() = default;

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;
	virtual void recompile(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
						   const std::vector<hiprtFuncNameSet>& func_name_sets = {},
						   bool silent										   = false,
						   bool use_cache									   = true) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_frame_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override {};

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_all_kernels() override;
	virtual std::map<std::string, std::shared_ptr<GPUKernel>> get_tracing_kernels() override;

private:
	enum class MaterialFamilyKernelType
	{
		PRIMARY_SHADING,
		SECONDARY_SHADING,
		DEFERRED_COMPLETION
	};

	unsigned int get_material_family_count(const GPUKernelCompilerOptions& compiler_options) const;
	bool uses_deferred_material_family_routing(const GPUKernelCompilerOptions& compiler_options) const;
	const std::string& get_primary_shading_kernel_name(unsigned int family_index) const;
	const std::string& get_secondary_shading_kernel_name(unsigned int family_index) const;
	const std::string& get_deferred_completion_kernel_name(unsigned int family_index) const;
	std::size_t get_nee_deferred_mis_context_byte_size();
	unsigned int get_shadow_work_capacity(unsigned int path_capacity) const;
	bool kernels_ready() const;
	bool resize_staging_buffers();
	void free_staging_buffers();
	WavefrontDataDevice get_wavefront_data_device();
	void reset_material_family_counts_async(oroStream_t stream);
	const std::string& get_material_family_kernel_name(MaterialFamilyKernelType kernel_type, unsigned int family_index) const;
	void launch_material_family_kernels_async(
		MaterialFamilyKernelType kernel_type, void** launch_args, unsigned int material_family_count, unsigned int block_count, bool overlap);

	int2_t m_render_resolution								   = make_int2(0, 0);
	bool m_staging_buffers_allocated						   = false;
	std::size_t m_allocated_ray_volume_state_byte_size		   = 0;
	std::size_t m_allocated_nee_deferred_mis_context_byte_size = 0;
	std::size_t m_nee_deferred_mis_context_byte_size		   = 0;
	bool m_nee_deferred_mis_context_byte_size_dirty			   = true;

	WavefrontDataHost<OrochiBuffer> m_wavefront_data;
	// Entry 0 is the render data of the kernels that run on the main stream.
	// Entry 1 + family_index is the render data of the kernels of that material family: each family gets
	// its own slice of the global BVH traversal stack buffer so that the families can run concurrently
	OrochiBuffer<HIPRTRenderData> m_render_data_host_pinned;
	OrochiBuffer<unsigned int> m_zero_host_pinned;

	// The shading and deferred completion kernels of the different material families run concurrently on their own streams.
	// On a RX 7900 XTX (white room, 1280x720) this went from 73 to 80 samples/s compared to launching them one after the other
	// on the main stream, with one HIP hardware queue per stream (see HIPRTOrochiCtx::set_HIP_hardware_queue_count())
	//
	// One stream per material family for the overlapped family kernels
	oroStream_t m_material_family_streams[KernelMaterialSpecializationCount] = {};
	// Recorded on the main stream before the family kernels so that the family streams wait for the previous work of the main stream
	oroEvent_t m_material_family_fork_event = nullptr;
	// Recorded on each family stream after its kernel so that the main stream waits for all the families before continuing
	oroEvent_t m_material_family_join_events[KernelMaterialSpecializationCount] = {};
};

#endif // #ifndef RENDERER_WAVEFRONT_RENDER_PASS_H
