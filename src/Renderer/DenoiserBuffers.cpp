/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Renderer/DenoiserBuffers.h"
#include "Renderer/GPURenderer.h"

float3* GPURendererDenoiserBuffers::map_normals_buffer()
{
	if (use_interop_AOVs)
		return m_normals_AOV_interop_buffer->map_no_error();
	else
		return m_normals_AOV_no_interop_buffer->get_device_pointer();
}

void GPURendererDenoiserBuffers::resize_normals_buffer(size_t new_element_count)
{
	if (use_interop_AOVs)
		m_normals_AOV_interop_buffer->resize(new_element_count);
	else
		m_normals_AOV_no_interop_buffer->resize(new_element_count);
}

void GPURendererDenoiserBuffers::unmap_normals_buffer()
{
	if (use_interop_AOVs)
		m_normals_AOV_interop_buffer->unmap();
}

ColorRGB32F* GPURendererDenoiserBuffers::map_albedo_buffer()
{
	if (use_interop_AOVs)
		return m_albedo_AOV_interop_buffer->map_no_error();
	else
		return m_albedo_AOV_no_interop_buffer->get_device_pointer();
}

void GPURendererDenoiserBuffers::resize_albedo_buffer(size_t new_element_count)
{
	if (use_interop_AOVs)
		m_albedo_AOV_interop_buffer->resize(new_element_count);
	else
		m_albedo_AOV_no_interop_buffer->resize(new_element_count);
}

void GPURendererDenoiserBuffers::unmap_albedo_buffer()
{
	if (use_interop_AOVs)
		m_albedo_AOV_interop_buffer->unmap();
}

void GPURendererDenoiserBuffers::set_use_interop_AOV_buffers(GPURenderer* renderer, bool use_interop)
{
	if (use_interop == use_interop_AOVs)
		// Nothing to change
		return;

	renderer->synchronize_kernel();

	use_interop_AOVs = use_interop;

	if (use_interop_AOVs)
	{
		m_normals_AOV_interop_buffer->resize(m_normals_AOV_no_interop_buffer->get_element_count());
		m_albedo_AOV_interop_buffer->resize(m_albedo_AOV_no_interop_buffer->get_element_count());

		m_normals_AOV_no_interop_buffer->free();
		m_albedo_AOV_no_interop_buffer->free();
	}
	else
	{
		m_normals_AOV_no_interop_buffer->resize(m_normals_AOV_interop_buffer->get_element_count());
		m_albedo_AOV_no_interop_buffer->resize(m_albedo_AOV_interop_buffer->get_element_count());

		m_normals_AOV_interop_buffer->free();
		m_albedo_AOV_interop_buffer->free();
	}
}
