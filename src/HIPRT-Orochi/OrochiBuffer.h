/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_BUFFER_H
#define OROCHI_BUFFER_H

#include "hiprt/hiprt.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Orochi/Orochi.h"
#include "UI/ImGui/ImGuiLogger.h"
#include "Utils/Utils.h"

extern ImGuiLogger g_imgui_logger;

template <typename T>
class OrochiBuffer
{
public:
	OrochiBuffer() : m_data_pointer(nullptr) {}
	OrochiBuffer(int element_count);
	OrochiBuffer(OrochiBuffer<T>&& other);
	~OrochiBuffer();

	void operator=(OrochiBuffer<T>&& other);

	void resize(int new_element_count, size_t type_size_override = 0);
	size_t get_element_count();

	T* get_device_pointer();
	T** get_pointer_address();

	std::vector<T> download_data() const;
	void download_data_async(void* out, oroStream_t stream) const;
	/**
	 * Uploads as many elements as returned by get_element_count from the data std::vector into the buffer.
	 * The given std::vector must therefore contain at least get_element_count() elements.
	 * 
	 * The overload using a void pointer reads sizeof(T) * get_element_count() bytes starting at
	 * the given pointer address. The given pointer must therefore provide a contiguous access
	 * to sizeof(T) * get_element_count() bytes of data
	 */
	void upload_data(const std::vector<T>& data);
	void upload_data(const void* data);

	/**
	 * Frees the buffer. No effect if already freed / not allocated yet
	 */
	void free();

private:
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
	free();
}

template <typename T>
void OrochiBuffer<T>::operator=(OrochiBuffer&& other)
{
	m_data_pointer = other.m_data_pointer;
	m_element_count = other.m_element_count;

	other.m_data_pointer = nullptr;
	other.m_element_count = 0;
}

template <typename T>
void OrochiBuffer<T>::resize(int new_element_count, size_t type_size_override)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));

	size_t buffer_size = type_size_override != 0 ? (type_size_override * new_element_count) : (sizeof(T) * new_element_count);
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), buffer_size));

	m_element_count = new_element_count;
}

template <typename T>
size_t OrochiBuffer<T>::get_element_count()
{
	return m_element_count;
}

template <typename T>
T* OrochiBuffer<T>::get_device_pointer()
{
	return m_data_pointer;
}

template <typename T>
T** OrochiBuffer<T>::get_pointer_address()
{
	return &m_data_pointer;
}

template <typename T>
std::vector<T> OrochiBuffer<T>::download_data() const
{
	if (!m_data_pointer)
		return std::vector<T>();

	std::vector<T> data(m_element_count);

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(data.data(), reinterpret_cast<oroDeviceptr>(m_data_pointer), sizeof(T) * m_element_count));

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

	oroMemcpyAsync(out, m_data_pointer, m_element_count * sizeof(T), oroMemcpyDeviceToHost, stream);
	//oroMemcpyDtoHAsync(out, m_data_pointer, m_element_count * sizeof(T), stream);
}

template <typename T>
void OrochiBuffer<T>::upload_data(const std::vector<T>& data)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_data_pointer), data.data(), sizeof(T) * m_element_count, oroMemcpyHostToDevice));
}

template <typename T>
void OrochiBuffer<T>::upload_data(const void* data)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpy(reinterpret_cast<oroDeviceptr>(m_data_pointer), data, sizeof(T) * m_element_count, oroMemcpyHostToDevice));
	else
		g_imgui_logger.add_line(ImGuiLoggerSeverity::IMGUI_LOGGER_ERROR, "Trying to upload data to an OrochiBuffer that hasn't been allocated yet!");
}

template <typename T>
void OrochiBuffer<T>::free()
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));

	m_element_count = 0;
	m_data_pointer = nullptr;
}

#endif
