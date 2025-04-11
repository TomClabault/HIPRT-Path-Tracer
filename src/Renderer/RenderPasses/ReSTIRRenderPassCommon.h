/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RESTIR_RENDER_PASS_COMMON_H
#define RESTIR_RENDER_PASS_COMMON_H

#include "HostDeviceCommon/ReSTIRSettingsHelper.h"
#include "Renderer/GPURenderer.h"

class ReSTIRRenderPassCommon
{
public:
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FUNCTION_NAME = "ReSTIR_Directional_Reuse_Compute";
	static constexpr const char* const DIRECTIONAL_REUSE_KERNEL_FILE = DEVICE_KERNELS_DIRECTORY "/ReSTIR/DirectionalReuseCompute.h";
	static constexpr const char* const DIRECTIONAL_REUSE_IS_RESTIR_GI_COMPILE_OPTION_NAME = "ComputingSpatialDirectionalReuseForReSTIRGI";

	static constexpr float AUTO_SPATIAL_RADIUS_RESOLUTION_PERCENTAGE = 0.025f;

	template <bool IsReSTIRGI>
	static void resize_common_buffers(GPURenderer* renderer, int new_width, int new_height,
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius,
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u,
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_reuse_buffer)
	{
		resize_decoupled_shading_reuse_buffer<IsReSTIRGI>(renderer, new_width, new_height, 
			decoupled_shading_reuse_buffer);
		resize_directional_reuse_buffers<IsReSTIRGI>(renderer, new_width, new_height,
			per_pixel_spatial_reuse_radius,
			per_pixel_spatial_reuse_direction_mask_u,
			per_pixel_spatial_reuse_direction_mask_ull);
	}

	template <bool IsReSTIRGI>
	static void resize_decoupled_shading_reuse_buffer(GPURenderer* renderer, int new_width, int new_height,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_reuse_buffer)
	{
		bool buffer_needed = (IsReSTIRGI
			? renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DO_SPATIAL_NEIGHBORS_DECOUPLED_SHADING)
			: renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_SPATIAL_NEIGHBORS_DECOUPLED_SHADING)) == KERNEL_OPTION_TRUE;

		ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(renderer->get_render_data());
		buffer_needed &= spatial_pass_settings.do_spatial_reuse_pass;

		if (buffer_needed)
			decoupled_shading_reuse_buffer.resize(new_width * new_height);
	}

	template <bool IsReSTIRGI>
	static void resize_directional_reuse_buffers(GPURenderer* renderer, int new_width, int new_height,
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius, 
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u, 
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull)
	{
		per_pixel_spatial_reuse_radius.resize(new_width * new_height);

		int bit_count = IsReSTIRGI 
			? renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT) 
			: renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT);

		if (bit_count <= 32)
			per_pixel_spatial_reuse_direction_mask_u.resize(new_width * new_height);
		else
			per_pixel_spatial_reuse_direction_mask_ull.resize(new_width * new_height);
	}
	
	template <bool IsReSTIRGI>
	static bool pre_render_update_common_buffers(const HIPRTRenderData& render_data, GPURenderer* renderer,
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius,
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u,
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_hits,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_total,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_buffer)
	{
		bool render_data_updated = false;

		render_data_updated |= pre_render_update_decoupled_shading_buffers<IsReSTIRGI>(render_data, renderer, 
			decoupled_shading_buffer);
		render_data_updated |= pre_render_update_directional_reuse_buffers<IsReSTIRGI>(render_data, renderer,
			per_pixel_spatial_reuse_radius,
			per_pixel_spatial_reuse_direction_mask_u,
			per_pixel_spatial_reuse_direction_mask_ull,
			spatial_reuse_statistics_hit_hits,
			spatial_reuse_statistics_hit_total);

		return render_data_updated;
	}

	template <bool IsReSTIRGI>
	static bool pre_render_update_decoupled_shading_buffers(const HIPRTRenderData& render_data, GPURenderer* renderer,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_buffer)
	{
		bool decoupled_shading_reuse_used = (IsReSTIRGI
			? renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_GI_DO_SPATIAL_NEIGHBORS_DECOUPLED_SHADING)
			: renderer->get_global_compiler_options()->get_macro_value(GPUKernelCompilerOptions::RESTIR_DI_DO_SPATIAL_NEIGHBORS_DECOUPLED_SHADING)) == KERNEL_OPTION_TRUE;
		decoupled_shading_reuse_used &= IsReSTIRGI ? render_data.render_settings.restir_gi_settings.common_spatial_pass.do_spatial_reuse_pass : render_data.render_settings.restir_di_settings.common_spatial_pass.do_spatial_reuse_pass;

		if (decoupled_shading_reuse_used)
		{
			if (decoupled_shading_buffer.get_element_count() == 0)
			{
				decoupled_shading_buffer.resize(renderer->m_render_resolution.x * renderer->m_render_resolution.y);

				return true;
			}
		}
		else
		{
			if (decoupled_shading_buffer.get_element_count() > 0)
			{
				decoupled_shading_buffer.free();

				return true;
			}
		}

		return false;
	}

	template <bool IsReSTIRGI>
	static bool pre_render_update_directional_reuse_buffers(const HIPRTRenderData& render_data, GPURenderer* renderer,
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius, 
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u, 
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_hits,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_total)
	{
		ReSTIRCommonSpatialPassSettings spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);
		const std::string& mask_bit_count_macro_name = IsReSTIRGI ? GPUKernelCompilerOptions::RESTIR_GI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT : GPUKernelCompilerOptions::RESTIR_DI_SPATIAL_DIRECTIONAL_REUSE_MASK_BIT_COUNT;
			
		// Allocating / deallocating the adaptive directional spatial reuse buffers if the feature
		// isn't used
		bool render_data_invalidated = false;
		if (spatial_pass_settings.do_adaptive_directional_spatial_reuse(render_data.render_settings.accumulate))
		{
			// Allocating the proper buffer whether or not we're using less than 32 bits per mask or more 
			if (renderer->get_global_compiler_options()->get_macro_value(mask_bit_count_macro_name) <= 32 && per_pixel_spatial_reuse_direction_mask_u.get_element_count() == 0)
			{
				per_pixel_spatial_reuse_direction_mask_u.resize(renderer->m_render_resolution.x * renderer->m_render_resolution.y);
				per_pixel_spatial_reuse_radius.resize(renderer->m_render_resolution.x * renderer->m_render_resolution.y);
				if (per_pixel_spatial_reuse_direction_mask_ull.get_element_count() > 0)
					per_pixel_spatial_reuse_direction_mask_ull.free();

				render_data_invalidated = true;
			}
			else if (renderer->get_global_compiler_options()->get_macro_value(mask_bit_count_macro_name) > 32 && per_pixel_spatial_reuse_direction_mask_ull.get_element_count() == 0)
			{
				per_pixel_spatial_reuse_direction_mask_ull.resize(renderer->m_render_resolution.x * renderer->m_render_resolution.y);
				per_pixel_spatial_reuse_radius.resize(renderer->m_render_resolution.x * renderer->m_render_resolution.y);
				if (per_pixel_spatial_reuse_direction_mask_u.get_element_count() > 0)
					per_pixel_spatial_reuse_direction_mask_u.free();

				render_data_invalidated = true;
			}
		}
		else
		{
			// We're not using the feature so we can free the buffers

			// Freeing the proper buffer depending on whether we use the 64 bits buffer or not
			if (per_pixel_spatial_reuse_direction_mask_u.get_element_count() > 0)
			{
				per_pixel_spatial_reuse_direction_mask_u.free();
				per_pixel_spatial_reuse_radius.free();

				render_data_invalidated = true;
			}
			else if (per_pixel_spatial_reuse_direction_mask_ull.get_element_count() > 0)
			{
				per_pixel_spatial_reuse_direction_mask_ull.free();
				per_pixel_spatial_reuse_radius.free();

				render_data_invalidated = true;
			}
		}

		// Also allocating / deallocating the buffers for the statistics
		if (spatial_pass_settings.compute_spatial_reuse_hit_rate)
		{
			if (spatial_reuse_statistics_hit_total.get_element_count() == 0)
			{
				spatial_reuse_statistics_hit_total.resize(1);
				spatial_reuse_statistics_hit_hits.resize(1);

				render_data_invalidated = true;
			}
		}
		else
		{
			// Freeing the buffers if the feature isn't used
			if (spatial_reuse_statistics_hit_total.get_element_count() > 0)
			{
				spatial_reuse_statistics_hit_total.free();
				spatial_reuse_statistics_hit_hits.free();

				render_data_invalidated = true;
			}
		}

		return render_data_invalidated;
	}

	template <bool IsReSTIRGI>
	static bool free_common_buffers(
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius,
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u,
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_hits,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_total,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_reuse_buffer)
	{
		bool render_data_invalidated = false;

		render_data_invalidated |= free_decoupled_shading_reuse_buffer<IsReSTIRGI>(decoupled_shading_reuse_buffer);
		render_data_invalidated |= free_directional_reuse_buffers<IsReSTIRGI>(
			per_pixel_spatial_reuse_radius,
			per_pixel_spatial_reuse_direction_mask_u,
			per_pixel_spatial_reuse_direction_mask_ull,
			spatial_reuse_statistics_hit_hits,
			spatial_reuse_statistics_hit_total);

		return render_data_invalidated;
	}

	template <bool IsReSTIRGI>
	static bool free_decoupled_shading_reuse_buffer(OrochiBuffer<ColorRGB32F>& decoupled_shading_reuse_buffer)
	{
		bool buffer_freed = false;

		if (decoupled_shading_reuse_buffer.get_element_count() > 0)
		{
			decoupled_shading_reuse_buffer.free();

			buffer_freed = true;
		}

		return buffer_freed;
	}

	template <bool IsReSTIRGI>
	static bool free_directional_reuse_buffers(
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius,
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u,
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_hits,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_total)
	{
		bool render_data_invalidated = false;

		if (per_pixel_spatial_reuse_direction_mask_u.get_element_count() > 0)
		{
			per_pixel_spatial_reuse_direction_mask_u.free();
			per_pixel_spatial_reuse_radius.free();

			render_data_invalidated = true;
		}

		if (per_pixel_spatial_reuse_direction_mask_ull.get_element_count() > 0)
		{
			per_pixel_spatial_reuse_direction_mask_ull.free();
			per_pixel_spatial_reuse_radius.free();

			render_data_invalidated = true;
		}

		if (spatial_reuse_statistics_hit_total.get_element_count() > 0)
		{
			spatial_reuse_statistics_hit_total.free();
			spatial_reuse_statistics_hit_hits.free();

			render_data_invalidated = true;
		}

		return render_data_invalidated;
	}

	template <bool IsReSTIRGI>
	static void update_render_data_common_buffers(HIPRTRenderData& render_data,
		OrochiBuffer<unsigned char>& per_pixel_spatial_reuse_radius,
		OrochiBuffer<unsigned int>& per_pixel_spatial_reuse_direction_mask_u,
		OrochiBuffer<unsigned long long int>& per_pixel_spatial_reuse_direction_mask_ull,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_hits,
		OrochiBuffer<unsigned long long int>& spatial_reuse_statistics_hit_total,
		OrochiBuffer<ColorRGB32F>& decoupled_shading_reuse_buffer)
	{
		ReSTIRCommonSpatialPassSettings& common_spatial_pass_settings = ReSTIRSettingsHelper::get_restir_spatial_pass_settings<IsReSTIRGI>(render_data);

		common_spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_u = per_pixel_spatial_reuse_direction_mask_u.get_device_pointer();
		common_spatial_pass_settings.per_pixel_spatial_reuse_directions_mask_ull = per_pixel_spatial_reuse_direction_mask_ull.get_device_pointer();
		common_spatial_pass_settings.per_pixel_spatial_reuse_radius = per_pixel_spatial_reuse_radius.get_device_pointer();

		common_spatial_pass_settings.spatial_reuse_hit_rate_total = reinterpret_cast<AtomicType<unsigned long long int>*>(spatial_reuse_statistics_hit_total.get_device_pointer());
		common_spatial_pass_settings.spatial_reuse_hit_rate_hits = reinterpret_cast<AtomicType<unsigned long long int>*>(spatial_reuse_statistics_hit_hits.get_device_pointer());

		common_spatial_pass_settings.decoupled_shading_reuse_buffer = decoupled_shading_reuse_buffer.get_device_pointer();
	}
};

#endif
