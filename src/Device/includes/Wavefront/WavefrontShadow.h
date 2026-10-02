/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_H
#define DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_H

#include "Device/includes/Hash.h"
#include "Device/includes/RayPayload.h"
#include "Device/includes/Wavefront/WavefrontShadowData.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"

HIPRT_DEVICE unsigned int pack_wavefront_shadow_rng_key(int bounce, unsigned int shadow_phase, unsigned int shadow_ordinal)
{
	return ((static_cast<unsigned int>(bounce) + 1u) << 8) | ((shadow_phase & 0xfu) << 4) | (shadow_ordinal & 0xfu);
}

HIPRT_DEVICE unsigned int make_wavefront_shadow_rng_seed(const HIPRTRenderData& render_data,
														 unsigned int path_index,
														 unsigned int packed_rng_key,
														 unsigned int stream_tag)
{
	unsigned int seed = wang_hash(static_cast<unsigned int>(render_data.render_settings.sample_number) + 1u);
	seed			  = wang_hash(seed ^ (path_index + 1u));
	seed			  = wang_hash(seed ^ packed_rng_key);

	return wang_hash(seed ^ stream_tag);
}

struct WavefrontShadowSink
{
	HIPRTRenderData& render_data;
	unsigned int path_index;
	ColorRGB32F throughput;
	int bounce;
	unsigned int shadow_phase;
	unsigned int emitted_count = 0;

	HIPRT_DEVICE WavefrontShadowSink(
		HIPRTRenderData& render_data, unsigned int path_index, ColorRGB32F throughput, int bounce, unsigned int shadow_phase = WAVEFRONT_SHADOW_PHASE_SHADE)
		: render_data(render_data), path_index(path_index), throughput(throughput), bounce(bounce), shadow_phase(shadow_phase)
	{
	}

	HIPRT_DEVICE unsigned int emit(float3_t origin,
								   float3_t direction,
								   float t_max,
								   int last_hit_primitive_index,
								   float3_t point_on_light,
								   ColorRGB32F contribution_if_visible,
								   unsigned int flags)
	{
		WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

		if (emitted_count == 0)
		{
			wavefront_data.path_shadow_throughputs[path_index] = throughput;
			wavefront_data.path_shadow_flags[path_index]	   = bounce == 0 ? WAVEFRONT_SHADOW_PATH_PRIMARY : 0u;
		}

		unsigned int work_items_per_path = wavefront_data.shadow_work_capacity / wavefront_data.path_capacity;
		if (emitted_count >= work_items_per_path)
		{
			hippt::atomic_fetch_add(wavefront_data.shadow_queue_overflow_count, 1u);
			return WAVEFRONT_INVALID_SHADOW_QUEUE_SLOT;
		}
		unsigned int queue_slot = emitted_count * wavefront_data.path_capacity + path_index;

		WavefrontShadowWork& work	  = wavefront_data.shadow_work_items[queue_slot];
		work.path_index				  = path_index;
		work.origin					  = origin;
		work.direction				  = direction;
		work.t_max					  = t_max;
		work.last_hit_primitive_index = last_hit_primitive_index;
		unsigned int packed_rng_key	  = pack_wavefront_shadow_rng_key(bounce, shadow_phase, emitted_count);

		// The low seven bits hold behavior flags. Keeping the deterministic RNG key in the
		// unused upper bits avoids a separate four-byte queue field and makes each item 64 bytes.
		work.flags					 = flags | (packed_rng_key << 7);
		work.contribution_if_visible = contribution_if_visible;
		work.point_on_light			 = point_on_light;

		emitted_count++;
		return queue_slot;
	}

	// The owning shade thread publishes once, after all emission for this path is complete.
	HIPRT_DEVICE void finish_emission()
	{
		if (has_work())
			render_data.wavefront_data.path_remaining_shadow_work_counts[path_index] = emitted_count;
	}

	HIPRT_DEVICE bool has_work() const
	{
		return emitted_count != 0;
	}

	HIPRT_DEVICE void mark_pending_finalize(const RayPayloadCommon& ray_payload, const Xorshift32Generator& random_number_generator)
	{
		WavefrontDataDevice& wavefront_data = render_data.wavefront_data;

		// This path will only accumulate its final color after shadow visibility is resolved.
		wavefront_data.path_ray_colors[path_index] = ray_payload.ray_color;
		wavefront_data.path_rng_states[path_index] = random_number_generator.m_state.seed;

		wavefront_data.path_shadow_flags[path_index] |= WAVEFRONT_SHADOW_PATH_PENDING_FINALIZE;
	}
};

#endif // #ifndef DEVICE_INCLUDES_WAVEFRONT_WAVEFRONT_SHADOW_H
