#ifndef OROCHI_BUFFER_H
#define OROCHI_BUFFER_H

#include "hiprt/hiprt.h"
#include "HIPRT-Orochi/orochi_utils.h"
#include "Orochi/Orochi.h"

template <typename T>
class OrochiBuffer
{
public:
	OrochiBuffer() : m_data_pointer(nullptr) {}
	OrochiBuffer(int element_count);

	void resize(int new_element_count);

	T* get_pointer();
	T** get_pointer_address();

	std::vector<T> download_pixels();
	void upload_pixels(std::vector<T>& data);

private:
	T* m_data_pointer;

	size_t m_element_count;
};

template <typename T>
OrochiBuffer<T>::OrochiBuffer(int element_count) : m_element_count(element_count)
{
	OROCHI_CHECK_ERROR(oroMalloc(reinterpret_cast<oroDeviceptr*>(&m_data_pointer), sizeof(T) * element_count));
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
T* OrochiBuffer<T>::get_pointer()
{
	return m_data_pointer;
}

template <typename T>
T** OrochiBuffer<T>::get_pointer_address()
{
	return &m_data_pointer;
}

template <typename T>
std::vector<T> OrochiBuffer<T>::download_pixels()
{
	if (!m_data_pointer)
		return std::vector<T>();

	std::vector<T> data(m_element_count);

	OROCHI_CHECK_ERROR(oroMemcpyDtoH(data.data(), reinterpret_cast<oroDeviceptr>(m_data_pointer), sizeof(T) * m_element_count));

	return data;
}

template <typename T>
void OrochiBuffer<T>::upload_pixels(std::vector<T>& data)
{
	void* data_pointer = data.data();
	if (m_data_pointer)
		OROCHI_CHECK_ERROR(oroMemcpyHtoD(reinterpret_cast<oroDeviceptr>(m_data_pointer), data_pointer, sizeof(T) * m_element_count));
}

#endif
