/* 
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_BUFFER_H
#define OROCHI_BUFFER_H

#include "hiprt/hiprt.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Orochi/Orochi.h"
#include "UI/DisplayView/DisplayTextureType.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"
#include "GL/glew.h"
#include "tracy/TracyOpenGL.hpp"

extern ImGuiLogger g_imgui_logger;

template <typename T>
class OrochiBuffer
{
public:
	using value_type = T;

	OrochiBuffer() : m_data_pointer(nullptr) {}
	OrochiBuffer(int element_count);
	OrochiBuffer(OrochiBuffer<T>&& other);
	~OrochiBuffer();

	void operator=(OrochiBuffer<T>&& other) noexcept;

	void memset_whole_buffer(T value);

	void resize(int new_element_count, size_t type_size_override = 0);
	void resize_host_pinned_mem(int new_element_count, size_t type_size_override = 0);
	size_t size() const;
	size_t get_byte_size() const;

	const T* get_device_pointer() const;
	T* get_device_pointer();

	const T* get_host_pinned_pointer() const;
	T* get_host_pinned_pointer();

	/**
	 * Returns a pointer to the device buffer but cast into an AtomicType
	 */
	const AtomicType<T>* get_atomic_device_pointer() const;
	AtomicType<T>* get_atomic_device_pointer();

	/**
	 * data() is just an alias for get_device_pointer()
	 */
	const T* data() const;
	T* data();

	bool is_allocated() const;

	/** 
	 * Static function for downloading from a device buffer when we
	 * only have the address of the buffer (and not the OrochiBuffer object)
	 */
	static std::vector<T> download_data(T* device_data_pointer, size_t element_count);
	std::vector<T> download_data() const;
	/**
	 * Download the data of the *whole* buffer directly to the given 'host_pointer'
	 * 
	 * The given 'host_pointer' is supposed to be pointing to an allocated memory block
	 * that is large enough to accomodate all the data of this buffer. Behavior is undefined
	 * if this is not the case
	 * 
	 * 'host_pointer' can also be the pointer returned by 'get_host_pinned_pointer()' if using host pinned
	 * memory
	 */
	void download_data_into(T* host_pointer) const;

	/**
	 * Downloads elements ['start_element_index', 'stop_element_index_excluded'[ from the buffer
	 */
	std::vector<T> download_data_partial(int start_element_index, int stop_element_index_excluded) const;
	void download_data_async(void* out, oroStream_t stream) const;


	static void upload_data(T* device_data_pointer, const std::vector<T>& data_to_upload, size_t element_count);
	static void upload_data(T* device_data_pointer, const T* data_to_upload, size_t element_count);
	/**
	 * Uploads as many elements as returned by size from the data std::vector into the buffer.
	 * The given std::vector must therefore contain at least size() elements.
	 * 
	 * The overload using a void pointer reads sizeof(T) * size() bytes starting at
	 * the given pointer address. The given pointer must therefore provide a contiguous access
	 * to sizeof(T) * size() bytes of data
	 */
	void upload_data(const std::vector<T>& data);
	void upload_data(const T* data);
	/**
	 * Uploads 'element_count' elmements from 'data' starting (it will be overriden) at element number 'start_index' in the buffer
	 */
	void upload_data_partial(int start_index, const T* data, size_t element_count);

	void unpack_to_GL_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type);

	/**
	 * Copies the data in 'other' to this buffer.
	 * 
	 * This copies the maximum amount of data from 'other' that can fit in this buffer
	 */
	void memcpy_from(const OrochiBuffer<T>& other);
	void memcpy_from(T* data_source, size_t element_count_to_copy);

	/**
	 * Frees the buffer. No effect if already freed / not allocated yet
	 */
	void free();

private:
	bool m_pinned_memory = false;

	T* m_data_pointer = nullptr;

	size_t m_element_count = 0;
};

template <typename T>
OrochiBuffer<T>::OrochiBuffer(int element_count) : m_element_count(element_count)
{
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), sizeof(T) * element_count));
}

template <typename T>
OrochiBuffer<T>::OrochiBuffer(OrochiBuffer<T>&& other)
{
	m_data_pointer = other.m_data_pointer;
	m_element_count = other.m_element_count;

	other.m_data_pointer = nullptr;
	other.m_element_count = 0;
}

template <typename T>
OrochiBuffer<T>::~OrochiBuffer()
{
	if (m_data_pointer)
		free();
}

template <typename T>
void OrochiBuffer<T>::operator=(OrochiBuffer&& other) noexcept
{
	if (m_data_pointer)
		free();

	m_data_pointer = other.m_data_pointer;
	m_element_count = other.m_element_count;

	other.m_data_pointer = nullptr;
	other.m_element_count = 0;
}

template<typename T>
inline void OrochiBuffer<T>::memset_whole_buffer(T value)
{
	if (m_data_pointer == nullptr)
	{
 		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to memset on an OrochiBuffer that hasn't been allocated yet!");
		return;
	}

	std::vector<T> data(m_element_count, value);

	upload_data(data);
}

template <typename T>
void OrochiBuffer<T>::resize(int new_element_count, size_t type_size_override)
{
	if (m_data_pointer)
		free();

	size_t buffer_size = type_size_override != 0 ? (type_size_override * new_element_count) : (sizeof(T) * new_element_count);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), buffer_size));

	m_element_count = new_element_count;
}

template <typename T>
void OrochiBuffer<T>::resize_host_pinned_mem(int new_element_count, size_t type_size_override)
{
	if (m_data_pointer)
		free();

	size_t buffer_size = type_size_override != 0 ? (type_size_override * new_element_count) : (sizeof(T) * new_element_count);
	OROCHI_CHECK_ERROR(oroHostMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), buffer_size, 0));

	m_element_count = new_element_count;
	m_pinned_memory = true;
}

template <typename T>
size_t OrochiBuffer<T>::size() const
{
	return m_element_count;
}

template <typename T>
size_t OrochiBuffer<T>::get_byte_size() const
{
	return m_element_count * sizeof(T);
}

template <typename T>
const T* OrochiBuffer<T>::get_device_pointer() const
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the device_pointer of an OrochiBuffer that hasn't been allocated!");

		return nullptr;
	}

	if (m_pinned_memory)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the device_pointer of an OrochiBuffer that has been allocated with host pinned memory. Pinned host memory doesn't have device pointers. Use get_host_pinned_pointer()");

		return nullptr;
	}
	else
		return m_data_pointer;
}

template <typename T>
T* OrochiBuffer<T>::get_device_pointer()
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the device_pointer of an OrochiBuffer that hasn't been allocated!");

		return nullptr;
	}

	if (m_pinned_memory)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the device_pointer of an OrochiBuffer that has been allocated with host pinned memory. Pinned host memory doesn't have device pointers. Use get_host_pinned_pointer()");

		return nullptr;
	}
	else
		return m_data_pointer;
}

template <typename T>
const AtomicType<T>* OrochiBuffer<T>::get_atomic_device_pointer() const
{
	return reinterpret_cast<AtomicType<T>*>(get_device_pointer());
}

template <typename T>
AtomicType<T>* OrochiBuffer<T>::get_atomic_device_pointer()
{
	return reinterpret_cast<AtomicType<T>*>(get_device_pointer());
}

template <typename T>
const T* OrochiBuffer<T>::get_host_pinned_pointer() const
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the host_pinned_pointer of an OrochiBuffer that hasn't been allocated!");

		return nullptr;
	}
	else if (!m_pinned_memory)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the host_pinned_pointer of an OrochiBuffer that has been allocated for the device! Use get_device_pointer() instead");

		return nullptr;
	}

	return m_data_pointer;
}

template <typename T>
T* OrochiBuffer<T>::get_host_pinned_pointer()
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the host_pinned_pointer of an OrochiBuffer that hasn't been allocated!");

		return nullptr;
	}
	else if (!m_pinned_memory)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Getting the host_pinned_pointer of an OrochiBuffer that has been allocated for the device! Use get_device_pointer() instead");

		return nullptr;
	}

	return m_data_pointer;
}


template <typename T>
const T* OrochiBuffer<T>::data() const
{
	return get_device_pointer();
}

template <typename T>
T* OrochiBuffer<T>::data()
{
	return get_device_pointer();
}

template <typename T>
bool OrochiBuffer<T>::is_allocated() const
{
	return m_data_pointer != nullptr;
}

// Static function
template <typename T>
std::vector<T> OrochiBuffer<T>::download_data(T* device_data_pointer, size_t element_count)
{
	if (!device_data_pointer)
		return std::vector<T>();

	std::vector<T> data(element_count);

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(data.data(), reinterpret_cast<oroDeviceptr>(device_data_pointer), sizeof(T) * element_count));

	return data;
}

template <typename T>
std::vector<T> OrochiBuffer<T>::download_data() const
{
	if (!m_data_pointer)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to download data from a non-allocated buffer!");

		return std::vector<T>();
	}

	std::vector<T> data(m_element_count);

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(data.data(), reinterpret_cast<oroDeviceptr>(m_data_pointer), sizeof(T) * m_element_count));

	return data;
}

template <typename T>
void OrochiBuffer<T>::download_data_into(T* host_pointer) const
{
	if (!m_data_pointer)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to download data into a host pinned buffer from a non-allocated buffer!");

		return;
	}

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(host_pointer, reinterpret_cast<oroDeviceptr>(m_data_pointer), sizeof(T) * m_element_count));
}

template<typename T>
inline std::vector<T> OrochiBuffer<T>::download_data_partial(int start_element_index, int stop_element_index_excluded) const
{
	if (!m_data_pointer)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to download data from a non-allocated buffer!");

		return std::vector<T>();
	}

	if (start_element_index == stop_element_index_excluded || stop_element_index_excluded < start_element_index)
		return std::vector<T>();

	size_t element_count = stop_element_index_excluded - start_element_index;
	std::vector<T> data(element_count);

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(data.data() + start_element_index, reinterpret_cast<oroDeviceptr>(m_data_pointer), sizeof(T) * element_count));

	return data;
}

template <typename T>
void OrochiBuffer<T>::download_data_async(void* out, oroStream_t stream) const 
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to download data async from a non-allocated buffer!");

		Utils::debugbreak();

		return;
	}

	OROCHI_CHECK_ERROR(oroMemcpyAsync(out, m_data_pointer, m_element_count * sizeof(T), oroMemcpyDeviceToHost, stream));
}

template <typename T>
void OrochiBuffer<T>::upload_data(T* device_data_pointer, const std::vector<T>& data_to_upload, size_t element_count)
{
	OrochiBuffer<T>::upload_data(device_data_pointer, data_to_upload.data(), element_count);
}

template <typename T>
void OrochiBuffer<T>::upload_data(T* device_data_pointer, const T* data_to_upload, size_t element_count)
{
	if (device_data_pointer && data_to_upload)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(device_data_pointer), data_to_upload, sizeof(T) * element_count, oroMemcpyHostToDevice));
}

template <typename T>
void OrochiBuffer<T>::upload_data(const std::vector<T>& data)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_data_pointer), data.data(), sizeof(T) * hippt::min(data.size(), m_element_count), oroMemcpyHostToDevice));
	else
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload data to an OrochiBuffer that hasn't been allocated yet!");
}

template <typename T>
void OrochiBuffer<T>::upload_data(const T* data)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_data_pointer), data, sizeof(T) * m_element_count, oroMemcpyHostToDevice));
	else
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload data to an OrochiBuffer that hasn't been allocated yet!");
}

template<typename T>
inline void OrochiBuffer<T>::upload_data_partial(int start_index, const T* data, size_t element_count)
{
	if (start_index > m_element_count)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload partial data to an OrochiBuffer starting at in an index that is larger than the buffer's size!");

		return;
	}

	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_data_pointer + start_index), data, sizeof(T) * element_count, oroMemcpyHostToDevice));
	else
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload partial data to an OrochiBuffer that hasn't been allocated yet!");
}

template <typename T>
void OrochiBuffer<T>::unpack_to_GL_texture(GLuint texture, GLint texture_unit, int width, int height, DisplayTextureType texture_type)
{
	glActiveTexture(texture_unit);
	glBindTexture(GL_TEXTURE_2D, texture);

	// Downloading the Orochi buffer and then uploading it back to the GPU.
	// Isn't that great code?
	//
	// The proper solution would be to use OpenGL Interop to copy the Orochi buffer
	// to the underlying array of the OpenGL texture. But it seems like OpenGL interop can only
	// do that for RGBA textures. But we're not stricly using RGBA textures here. The template type
	// could be anything really and it at least doesn't work with float3 types because float3 are RGB,
	// not RGBA and again, OpenGL Interop throws an error at 'oroGraphicsGLRegisterImage' for RGB
	// textures.
	//
	// So to fix this, we could use an RGBA OpenGL texture in place of RGB. But then, in the case of 
	// world-space normals buffer for example, we have to convert our float3 normals to float4 to upload
	// to the RGBA texture. And that conversion would be expensive (and require memory)
	//
	// We could also just use float4 data all the way for the normals. We wouldn't have any conversion to do.
	// But we would have a conversion to perform before denoising and so the issues would be the same
	//
	// So maybe there is another solution besides the RGBA OpenGL Interop but too lazy, this is an unlikely
	// code path in the application anyways
	std::vector<T> data = download_data();
	glTexImage2D(GL_TEXTURE_2D, 0, texture_type.get_gl_internal_format(), width, height, 0, texture_type.get_gl_format(), texture_type.get_gl_type(), data.data());

	//oroGraphicsResource_t graphics_resource = nullptr;
	//OROCHI_CHECK_ERROR(oroGraphicsGLRegisterImage(&graphics_resource, texture, GL_TEXTURE_2D, oroGraphicsRegisterFlagsWriteDiscard));

	//// Map the OpenGL texture for CUDA/HIP access
	//OROCHI_CHECK_ERROR(oroGraphicsMapResources(1, &graphics_resource, 0));

	//// Access the CUDA/HIP array used by the OpenGL texture under the hood
	//oroArray_t array = nullptr;
	//	OROCHI_CHECK_ERROR(oroGraphicsSubResourceGetMappedArray(&array, graphics_resource, 0, 0));

	//// Copy data from the CUDA buffer to the CUDA array
	//	OROCHI_CHECK_ERROR(oroMemcpy2DToArray(array, 0, 0, m_data_pointer, width * texture_type.sizeof_type(), width * texture_type.sizeof_type(), height, oroMemcpyDeviceToDevice));

	//// Unmap the OpenGL texture
	//OROCHI_CHECK_ERROR(oroGraphicsUnmapResources(1, &graphics_resource, 0));
}

template<typename T>
inline void OrochiBuffer<T>::memcpy_from(const OrochiBuffer<T>& other)
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to memcpy_from() into an OrochiBuffer that hasn't been allocated yet!");

		return;
	}

	size_t size_to_copy = std::min(other.m_element_count, m_element_count);
	OROCHI_CHECK_ERROR(oroMemcpy(m_data_pointer, other.get_device_pointer(), size_to_copy * sizeof(T), oroMemcpyDeviceToDevice));
}

template<typename T>
inline void OrochiBuffer<T>::memcpy_from(T* data_source, size_t element_count_to_copy)
{
	if (m_data_pointer == nullptr)
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to memcpy_from() into an OrochiBuffer that hasn't been allocated yet!");

		return;
	}

	OROCHI_CHECK_ERROR(oroMemcpy(m_data_pointer, data_source, element_count_to_copy * sizeof(T), oroMemcpyDeviceToDevice));
}

template <typename T>
void OrochiBuffer<T>::free()
{
	if (m_data_pointer)
	{
		if (m_pinned_memory)
			OROCHI_CHECK_ERROR(oroHostFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));
		else
			OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));
	}
	else
	{
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Freeing an OrochiBuffer buffer that hasn't been initialized (or has been freed already)!");

		return;
	}

	m_element_count = 0;
	m_data_pointer = nullptr;
	m_pinned_memory = false;
}

#endif
