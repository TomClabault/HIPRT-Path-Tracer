/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef KERNELS_WAVEFRONT_TRACE_SHADOW_RAYS_H
#define KERNELS_WAVEFRONT_TRACE_SHADOW_RAYS_H

#include "Device/includes/Intersect.h"
#include "Device/includes/LightSampling/LearningToClusterEnvmapSelection.h"
#include "Device/includes/Wavefront/WavefrontCommon.h"
#include "Device/includes/Wavefront/WavefrontShadow.h"

HIPRT_DEVICE void wavefront_update_shadow_training(
	HIPRTRenderData& render_data, unsigned int queue_slot, unsigned int path_index, unsigned int flags, bool visible)
{
	if ((flags & WAVEFRONT_SHADOW_UPDATE_LEARNING_TO_CLUSTER) != 0)
	{
		WavefrontLearningToClusterShadowSidecar& sidecar = render_data.wavefront_data.learning_to_cluster_shadow_sidecars[queue_slot];

		if (visible)
		{
			if (sidecar.spatial_training_sample_index < render_data.kd_tree_device.core.training_sample_capacity)
				render_data.kd_tree_device.core.training_samples[sidecar.spatial_training_sample_index].spatial_radiance_weight =
					sidecar.spatial_radiance_weight_if_visible;

			if (sidecar.learning_to_cluster_sample_index < render_data.kd_tree_device.learning_to_cluster.training_sample_capacity)
			{
				IlluminationAwareKDTreeLearningToClusterTrainingSample& training_sample =
					render_data.kd_tree_device.learning_to_cluster.training_samples[sidecar.learning_to_cluster_sample_index];
				training_sample.q_reward			 = sidecar.cluster_reward_if_visible;
				training_sample.variance_observation = sidecar.cluster_reward_if_visible;
			}
		}
	}

	// The thread that traces the shadow rays of a path is the only one that writes the statistics of its pixel during this pass
	if ((flags & WAVEFRONT_SHADOW_UPDATE_PIXEL_ENVMAP_STATISTICS) != 0 && visible)
	{
		const WavefrontLearningToClusterShadowSidecar& sidecar = render_data.wavefront_data.learning_to_cluster_shadow_sidecars[queue_slot];

		bool is_envmap = (flags & WAVEFRONT_SHADOW_IS_ENVMAP) != 0;
		add_learning_to_cluster_pixel_envmap_statistic_observation(render_data, path_index, is_envmap, sidecar.pixel_envmap_statistic_observation_if_visible);
	}
}

HIPRT_DEVICE void wavefront_commit_shadow_contributions(HIPRTRenderData& render_data, unsigned int path_index, ColorRGB32F contribution)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	unsigned int shadow_flags			= wavefront_data.path_shadow_flags[path_index];

	ColorRGB32F final_ray_color = wavefront_data.path_ray_colors[path_index] + contribution;

	if ((shadow_flags & WAVEFRONT_SHADOW_PATH_PENDING_FINALIZE) != 0)
	{
		Xorshift32Generator random_number_generator(wavefront_data.path_rng_states[path_index]);
		int x = path_index % render_data.render_settings.render_resolution.x;
		int y = path_index / render_data.render_settings.render_resolution.x;

		wavefront_finalize_color(render_data, path_index, x, y, final_ray_color, random_number_generator);
	}
	else
		// Only continuing paths need the updated persistent path color.
		wavefront_data.path_ray_colors[path_index] = final_ray_color;
}

HIPRT_DEVICE void wavefront_trace_shadow_queue(HIPRTRenderData& render_data, unsigned int path_slot, unsigned int path_stride)
{
	WavefrontDataDevice& wavefront_data = render_data.wavefront_data;
	unsigned int work_items_per_path	= wavefront_data.shadow_work_capacity / wavefront_data.path_capacity;

	for (; path_slot < wavefront_data.path_capacity; path_slot += path_stride)
	{
		unsigned int path_work_count = wavefront_data.path_remaining_shadow_work_counts[path_slot];
		if (path_work_count == 0)
			continue;

		if (path_work_count > work_items_per_path)
			path_work_count = work_items_per_path;

		// Sum split samples, keeping only one contribution live across traversal.
		ColorRGB32F contribution(0.0f);

		for (unsigned int work_ordinal = 0; work_ordinal < path_work_count; work_ordinal++)
		{
			unsigned int queue_slot		 = work_ordinal * wavefront_data.path_capacity + path_slot;
			WavefrontShadowWork work	 = wavefront_data.shadow_work_items[queue_slot];
			unsigned int packed_rng_key	 = work.flags >> WAVEFRONT_SHADOW_RNG_KEY_SHIFT;
			unsigned int shadow_rng_seed = make_wavefront_shadow_rng_seed(render_data, work.path_index, packed_rng_key, 0x53484457u);
			Xorshift32Generator random_number_generator(shadow_rng_seed);

			hiprtRay shadow_ray;
			shadow_ray.origin	 = work.origin;
			shadow_ray.direction = work.direction;

			bool occluded = false;

			if ((work.flags & WAVEFRONT_SHADOW_USE_NEE_PLUS_PLUS) != 0)
			{
				NEEPlusPlusContext nee_plus_plus_context;
				nee_plus_plus_context.shaded_point	 = shadow_ray.origin;
				nee_plus_plus_context.point_on_light = work.point_on_light;
				nee_plus_plus_context.envmap		 = (work.flags & WAVEFRONT_SHADOW_IS_ENVMAP) != 0;

				occluded = evaluate_shadow_ray_nee_plus_plus(render_data, shadow_ray, work.t_max, work.last_hit_primitive_index, nee_plus_plus_context,
															 random_number_generator);
			}
			else
				occluded = evaluate_shadow_ray_occluded(render_data, shadow_ray, work.t_max, work.last_hit_primitive_index, random_number_generator);

			bool visible = !occluded;
			if (visible)
				contribution += work.contribution_if_visible;

			wavefront_update_shadow_training(render_data, queue_slot, work.path_index, work.flags, visible);
		}

		wavefront_data.path_remaining_shadow_work_counts[path_slot] = 0;
		wavefront_commit_shadow_contributions(render_data, path_slot, contribution);
	}
}

#ifdef __KERNELCC__
extern "C"
{
	HIPRT_DEVICE __constant__ unsigned char WAVEFRONT_TRACE_SHADOW_RENDER_DATA[sizeof(HIPRTRenderData)];
}

GLOBAL_KERNEL_SIGNATURE(void) __launch_bounds__(64) WavefrontTraceShadowRays()
{
	HIPRTRenderData& render_data = *reinterpret_cast<HIPRTRenderData*>(WAVEFRONT_TRACE_SHADOW_RENDER_DATA);
	unsigned int queue_slot		 = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int queue_stride	 = blockDim.x * gridDim.x;

	wavefront_trace_shadow_queue(render_data, queue_slot, queue_stride);
}
#else
GLOBAL_KERNEL_SIGNATURE(void) inline WavefrontTraceShadowRays(HIPRTRenderData render_data, unsigned int queue_slot)
{
	unsigned int work_count = *render_data.wavefront_data.shadow_work_count;
	if (work_count > render_data.wavefront_data.shadow_work_capacity)
		work_count = render_data.wavefront_data.shadow_work_capacity;

	wavefront_trace_shadow_queue(render_data, queue_slot, work_count);
}
#endif

#endif // #ifndef KERNELS_WAVEFRONT_TRACE_SHADOW_RAYS_H
