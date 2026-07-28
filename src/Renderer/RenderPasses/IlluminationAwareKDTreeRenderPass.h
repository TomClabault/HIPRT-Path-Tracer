/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H

#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstdint>

class IlluminationAwareKDTreeRenderPass : public RenderPass
{
public:
	static const std::string ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME;
	static const std::string RESET_TREE_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID;
	static const std::string RESET_BATCH_STATISTICS_KERNEL_ID;
	static const std::string EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID;
	static const std::string REPLAY_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID;
	static const std::string MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID;
	static const std::string PROMOTE_GUIDING_CELLS_KERNEL_ID;

	static constexpr int INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY = 2000000;

public:
	IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_render_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	void ensure_all_lookahead_cell_levels(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;
	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

	int& get_split_iterations_per_SPP();
	int& get_training_sample_buffer_capacity();
	void mark_buffers_need_reallocation();

	bool& get_frozen_tree();

	std::size_t get_vram_usage_bytes() const;

private:
	// DEBUG
	void run_mark_guiding_cells_for_splitting_debug_check();
	void run_promote_guiding_cells_debug_check();
	void run_debug_check(const GPUKernelCompilerOptions& compiler_options);
	void print_current_tree_debug_info(std::ostream& output);
	// DEBUG

	IlluminationAwareKDTreeSubdivisionMode m_subdivision_mode = IlluminationAwareKDTreeSubdivisionMode::DISABLED;

	bool m_frozen_tree				 = false;
	bool m_buffers_need_reallocation = true;
	IlluminationAwareKDTreeDataHost<OrochiBuffer> m_illumination_aware_kd_tree;

	// How many times to:
	//	for (int split; split < m_split_iterations; split++)
	//	{
	//		- Create lookahead nodes below guiding cells
	//		- Replay training samples to the newly created lookahead nodes
	//		- Split guiding cells that have been marked for splitting
	//		- Accumulate the statistics of the newly created lookahead nodes into their history
	//	}
	// per each SPP
	int m_split_iterations_per_SPP		  = 1;
	int m_training_sample_buffer_capacity = INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY;

	bool m_lookahead_frontier_initialized	  = false;
	bool m_current_frontier_uses_first_buffer = true;

	unsigned int m_next_creation_tag = 0;

	bool m_mark_guiding_cells_debug_check_done	  = false;
	bool m_promote_guiding_cells_debug_check_done = false;
};

#endif
