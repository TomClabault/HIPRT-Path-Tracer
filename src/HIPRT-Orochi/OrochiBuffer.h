/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef OROCHI_BUFFER_H
#define OROCHI_BUFFER_H

#include "hiprt/hiprt.h"
#include "HIPRT-Orochi/HIPRTOrochiUtils.h"
#include "Orochi/Orochi.h"

template <typename T>
class OrochiBuffer
{
public:
	OrochiBuffer() : m_data_pointer(nullptr) {}
	OrochiBuffer(int element_count);
	OrochiBuffer(OrochiBuffer<T>&& other);
	~OrochiBuffer();

	void operator=(OrochiBuffer<T>&& other);

	void resize(int new_element_count);
	size_t get_element_count();

	T* get_device_pointer();
	T** get_pointer_address();

	std::vector<T> download_data() const;
	void upload_data(const std::vector<T>& data);
	void upload_data(const void* data);

	void destroy();

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
	destroy();
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
void OrochiBuffer<T>::resize(int new_element_count)
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));

	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), sizeof(T) * new_element_count));

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
		std::cerr << "Trying to upload data to an OrochiBuffer that hasn't been allocated yet!" << std::endl;
}

template <typename T>
void OrochiBuffer<T>::destroy()
{
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroFree(reinterpret_cast<oroDeviceptr>(m_data_pointer)));
}

#endif
