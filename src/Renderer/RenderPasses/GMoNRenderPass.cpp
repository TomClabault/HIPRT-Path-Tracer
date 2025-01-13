/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "GMoNRenderPass.h"

bool GMoNRenderPass::use_gmon()
{
	return m_gmon.use_gmon;
}

bool GMoNRenderPass::update(HIPRTRenderData& render_data)
{
	int2 render_resolution = render_data.render_settings.render_resolution;

	if (m_gmon.use_gmon)
	{
		if (m_gmon.current_resolution.x != render_resolution.x || m_gmon.current_resolution.y != render_resolution.y)
		{
			m_gmon.resize_sets(render_resolution.x, render_resolution.y);
			m_gmon.resize_interop(render_resolution.x, render_resolution.y);
			render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;

			return true;
		}
		else
		{
			render_data.buffers.gmon_estimator.next_set_to_accumulate++;
			if (render_data.buffers.gmon_estimator.next_set_to_accumulate == m_gmon.number_of_sets)
				render_data.buffers.gmon_estimator.next_set_to_accumulate = 0;
		}
	}
	else
	{
		m_gmon.free();

		return true;
	}

	return false;
}

std::shared_ptr<OpenGLInteropBuffer<ColorRGB32F>> GMoNRenderPass::get_result_framebuffer()
{
	return m_gmon.result_framebuffer;
}

ColorRGB32F* GMoNRenderPass::get_sets_buffers_device_pointer()
{
	return m_gmon.sets.get_device_pointer();
}

void GMoNRenderPass::resize_interop_buffers(unsigned int new_width, unsigned int new_height)
{
	if (m_gmon.use_gmon)
		m_gmon.result_framebuffer->resize(new_width * new_height);
}

void GMoNRenderPass::resize_non_interop_buffers(unsigned int new_width, unsigned int new_height)
{
	if (m_gmon.use_gmon)
		m_gmon.resize_sets(new_width, new_height);
}

ColorRGB32F* GMoNRenderPass::map_result_framebuffer()
{
	return m_gmon.map_result_framebuffer();
}

void GMoNRenderPass::unmap_result_framebuffer()
{
	if (m_gmon.use_gmon)
		m_gmon.result_framebuffer->unmap();
}
