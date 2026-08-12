/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_NISML_RENDER_PASS_H
#define RENDERER_NISML_RENDER_PASS_H

#include "HostDeviceCommon/KernelOptions/NeuralImportanceSamplingManyLightsOptions.h"
#include "Renderer/CPUGPUCommonDataStructures/Neural/MLPDataHost.h"
#include "Renderer/CPUGPUCommonDataStructures/Neural/NISMLPositionLearnableDenseGridDataHost.h"
#include "Renderer/RenderPasses/RenderPass.h"

#include "Renderer/CPUGPUCommonDataStructures/Neural/NISMLDataHost.h"

#include <cstddef>

struct NISMLVRAMUsage
{
	std::size_t neurons_biases				= 0;
	std::size_t gradient_biases				= 0;
	std::size_t connection_weights			= 0;
	std::size_t connection_weights_fp16		= 0;
	std::size_t gradient_weights			= 0;
	std::size_t training_sample_count		= 0;
	std::size_t adam_weights_means			= 0;
	std::size_t adam_weights_variances		= 0;
	std::size_t adam_biases_means			= 0;
	std::size_t adam_biases_variances		= 0;
	std::size_t train_activations			= 0;
	std::size_t training_records			= 0;
	std::size_t training_record_count		= 0;
	std::size_t grid_features				= 0;
	std::size_t grid_features_fp16			= 0;
	std::size_t grid_gradient_features		= 0;
	std::size_t grid_adam_feature_means		= 0;
	std::size_t grid_adam_feature_variances = 0;

	std::size_t get_total_bytes() const
	{
		std::size_t mlp_and_records_bytes = neurons_biases + gradient_biases + connection_weights + connection_weights_fp16 + gradient_weights +
											training_sample_count + adam_weights_means + adam_weights_variances + adam_biases_means + adam_biases_variances +
											train_activations + training_records + training_record_count;

		return mlp_and_records_bytes + grid_features + grid_features_fp16 + grid_gradient_features + grid_adam_feature_means + grid_adam_feature_variances;
	}
};

class NISMLRenderPass : public RenderPass
{
public:
	static const std::string NISML_RENDER_PASS_NAME;
	static const std::string NISML_TRAIN;
	static const std::string NISML_OPTIMIZE;
	static const std::string NISML_GRID_OPTIMIZE;

	NISMLRenderPass(GPURenderer* renderer, std::shared_ptr<GPUKernelCompilerOptions> options);

	virtual bool pre_render_compilation_check(std::shared_ptr<HIPRTOrochiCtx>& hiprt_orochi_ctx,
											  const std::vector<hiprtFuncNameSet>& func_name_sets,
											  bool silent,
											  bool use_cache) override;

	virtual void resize(unsigned int new_width, unsigned int new_height) override;

	virtual bool pre_sample_update(float delta_time) override;
	virtual bool launch_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;
	virtual void post_sample_update_async(HIPRTRenderData& render_data, GPUKernelCompilerOptions& compiler_options) override;

	virtual void update_render_data() override;
	virtual void reset(bool reset_by_camera_movement) override;

	virtual bool is_render_pass_used(const GPUKernelCompilerOptions& compiler_options) const override;

	float& get_training_record_percentage();
	int& get_training_spp();
	int& get_training_record_buffer_capacity();
	float& get_adam_learning_rate();

	NISMLVRAMUsage get_vram_usage_breakdown() const;
	std::size_t get_vram_usage_bytes() const;

private:
	bool pre_render_update();

	void print_train_profile(unsigned int training_record_count);

	MLPDataHost<OrochiBuffer, NeuralImportanceSamplingMLP> m_mlp;
	NISMLPositionLearnableDenseGridDataHost<OrochiBuffer> m_position_learnable_dense_grid;
	NISMLDataHost<OrochiBuffer> m_nisml_data;

	OrochiBuffer<NISMLTrainProfileRecord> m_train_profile_records;

	unsigned int m_adam_step			  = 0;
	float m_training_record_percentage	  = 15.0f;
	int m_training_spp					  = 0;
	int m_training_record_buffer_capacity = NISMLDataHost<OrochiBuffer>::NISML_TRAINING_BATCH_SIZE;
	float m_adam_learning_rate			  = 0.03f;
};

#endif
