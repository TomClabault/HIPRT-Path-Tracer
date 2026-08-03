/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H
#define RENDERER_ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_H

#include "Renderer/CPUGPUCommonDataStructures/IlluminationAwareKDTreeDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include <cstdint>

struct IlluminationAwareKDTreeVRAMUsage
{
	std::size_t nodes		= 0;
	std::size_t node_bounds = 0;
	std::size_t node_count	= 0;

	std::size_t active_guiding_nodes	  = 0;
	std::size_t active_guiding_node_count = 0;
	std::size_t needs_split				  = 0;
	std::size_t light_clustering_count	  = 0;

	std::size_t current_frontier	   = 0;
	std::size_t current_frontier_count = 0;
	std::size_t next_frontier		   = 0;
	std::size_t next_frontier_count	   = 0;

	std::size_t training_samples						  = 0;
	std::size_t training_sample_count					  = 0;
	std::size_t learning_to_cluster_training_samples	  = 0;
	std::size_t learning_to_cluster_training_sample_count = 0;

	std::size_t batch_signatures		= 0;
	std::size_t history_signatures		= 0;
	std::size_t batch_spatial_moments	= 0;
	std::size_t history_spatial_moments = 0;

	std::size_t initial_light_cut_node_indices		  = 0;
	std::size_t light_cluster_node_indices			  = 0;
	std::size_t light_cluster_statistics			  = 0;
	std::size_t light_cluster_batch_statistics		  = 0;
	std::size_t light_clustering_data				  = 0;
	std::size_t light_clustering_batch_sample_counts  = 0;
	std::size_t representative_shading_contexts		  = 0;
	std::size_t representative_shading_context_states = 0;

	std::size_t get_total_bytes() const
	{
		return nodes + node_bounds + node_count + active_guiding_nodes + active_guiding_node_count + needs_split + light_clustering_count + current_frontier +
			   current_frontier_count + next_frontier + next_frontier_count + training_samples + training_sample_count + learning_to_cluster_training_samples +
			   learning_to_cluster_training_sample_count + batch_signatures + history_signatures + batch_spatial_moments + history_spatial_moments +
			   initial_light_cut_node_indices + light_cluster_node_indices + light_cluster_statistics + light_cluster_batch_statistics + light_clustering_data +
			   light_clustering_batch_sample_counts + representative_shading_contexts + representative_shading_context_states;
	}
};

class IlluminationAwareKDTreeRenderPass : public RenderPass
{
public:
	static const std::string ILLUMINATION_AWARE_KD_TREE_RENDER_PASS_NAME;
	static const std::string RESET_TREE_KERNEL_ID;
	static const std::string INITIALIZE_ROOT_LIGHT_CLUSTERING_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string ACCUMULATE_BATCH_STATISTICS_INTO_HISTORY_KERNEL_ID;
	static const std::string RESET_BATCH_KD_TREE_AND_LIGHT_CLUSTERING_STATISTICS_KERNEL_ID;
	static const std::string EXPAND_ONE_LOOKAHEAD_LEVEL_KERNEL_ID;
	static const std::string REPLAY_TRAINING_SAMPLES_KERNEL_ID;
	static const std::string INITIALIZE_CREATED_NODE_HISTORY_KERNEL_ID;
	static const std::string MARK_GUIDING_CELLS_FOR_SPLITTING_KERNEL_ID;
	static const std::string PROMOTE_GUIDING_CELLS_KERNEL_ID;

	static constexpr int INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY = 2000000;

public:
	IlluminationAwareKDTreeRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual void resize(unsigned int new_width, unsigned int new_height) override;
	virtual bool pre_sample_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	void ensure_all_lookahead_cell_levels(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options);

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;
	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

	int& get_split_iterations_per_SPP();
	bool& get_auto_split_iterations_per_SPP();
	int& get_training_sample_buffer_capacity();

	std::size_t get_current_node_buffer_capacity() const;
	std::size_t get_current_node_count() const;
	std::size_t get_current_guiding_node_count() const;

	void mark_buffers_need_reallocation();

	bool& get_frozen_tree();

	/**
	 * If this function is updated, IlluminationAwareKDTreeVRAMUsage::get_total_bytes() should also be updated and the ImGui UI should also be updated to keep
	 * the tooltip up to date
	 */
	IlluminationAwareKDTreeVRAMUsage get_vram_usage_breakdown() const;
	std::size_t get_vram_usage_bytes() const;

private:
	bool m_frozen_tree				 = false;
	bool m_buffers_need_reallocation = true;

	IlluminationAwareKDTreeUserSettings m_user_settings;
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
	//
	// Higher number subdivide faster but is more expensive
	int m_split_iterations_per_SPP = 3;
	// If true, the number of split iterations per SPP will be automatically adjusted based on the current SPP for efficiency
	bool m_auto_split_iterations_per_SPP  = true;
	int m_training_sample_buffer_capacity = INITIAL_TRAINING_SAMPLE_BUFFER_CAPACITY;

	std::size_t m_cached_current_node_count			= 1;
	std::size_t m_cached_current_guiding_node_count = 1;

	bool m_lookahead_frontier_initialized	  = false;
	bool m_current_frontier_uses_first_buffer = true;

	unsigned int m_next_creation_tag = 0;

	bool m_mark_guiding_cells_debug_check_done	  = false;
	bool m_promote_guiding_cells_debug_check_done = false;
};

#endif
