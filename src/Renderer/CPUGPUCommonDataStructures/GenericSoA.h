/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GENERIC_SOA_H
#define RENDERER_GENERIC_SOA_H

#include <atomic>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Renderer/CPUGPUCommonDataStructures/GenericAtomicType.h"
#include "Renderer/CPUGPUCommonDataStructures/GenericFP16Type.h"

// Helper to detect std::atomic<...>
//
// std::false_type and std::true_type are structures that
// have ::value equal to 'false' or ::value equal to 'true' respectively
//
// By inheriting from std::false_type or std::true_type, we can check at compile time
// what's our ::value and use a constexpr if() on that
template <typename T>
struct IsStdAtomic : std::false_type
{
};

template <typename U>
struct IsStdAtomic<std::atomic<U>> : std::true_type
{
};

namespace GenericSoAHelpers
{
	template <template <typename> class BufferContainer, typename T>
	std::size_t get_byte_size(const BufferContainer<T>& buffer)
	{
		return buffer.size() * sizeof(typename BufferContainer<T>::value_type);
	}
} // namespace GenericSoAHelpers

/**
 * Can be used to create a structure of arrays for multiple buffers of different types.
 *
 * For example, to declare an SoA of 3 buffers: float, float and int, this can be used as:
 *
 * GenericSoA<std::vector, float, float, int> if the SoA is meant to be used on the CPU (std::vector)
 * GenericSoA<std::vector, float, float, int> if the SoA is meant to be used on the GPU (OrochiBuffer)
 *
 * The 'Container' type must support the following operations:
 *      - resize(int new_element_count) -> resizes the container to hold new_element_count elements
 *      - size() -> returns the number of elements in the container
 */
template <template <typename> class Container, typename... Types>
struct GenericSoA
{
	template <typename T>
	using BufferTypeFromVariable = typename std::decay_t<T>::value_type;

	template <int bufferIndex>
	using BufferTypeFromIndex = typename std::tuple_element<bufferIndex, std::tuple<Container<Types>...>>::type::value_type;

	using IsCPUBuffer = std::is_same<Container<BufferTypeFromIndex<0>>, std::vector<BufferTypeFromIndex<0>>>;

	void resize(std::size_t new_element_count, std::unordered_set<int> excluded_buffer_indices = {})
	{
		m_maximum_size = new_element_count;

		// Applies resize(new_element_count) on each buffer in the tuple and handles the excluded buffers
		resize_with_exclusions_internal(new_element_count, excluded_buffer_indices, std::index_sequence_for<Types...>{});
	}

	template <int bufferIndex>
	void resize_one_buffer(std::size_t new_element_count)
	{
		if (new_element_count > m_maximum_size)
			m_maximum_size = new_element_count;

		resize_buffer_internal(get_buffer<bufferIndex>(), new_element_count);
	}

	std::size_t get_byte_size() const
	{
		std::size_t total = 0;

		// For each container, add sizeof(value_type) * size()
		std::apply([&](const auto&... buffer) { ((total += GenericSoAHelpers::get_byte_size(buffer)), ...); }, buffers);

		return total;
	}

	std::size_t maximum_size() const
	{
		return m_maximum_size;
	}

	template <int bufferIndex>
	void memset_buffer(BufferTypeFromIndex<bufferIndex> memset_value)
	{
		if constexpr (IsCPUBuffer::value)
		{
			if constexpr (IsStdAtomic<BufferTypeFromIndex<bufferIndex>>::value)
			{
				// For atomic types, we have to store into them with a loop because they do not have an =operator()
				// so we can't use std::fill
				for (auto& value : get_buffer<bufferIndex>())
					value.store(memset_value);
			}
			else
				std::fill(get_buffer<bufferIndex>().begin(), get_buffer<bufferIndex>().end(), memset_value);
		}
		else
		{
			std::vector<BufferTypeFromIndex<bufferIndex>> data(get_buffer<bufferIndex>().size(), memset_value);
			get_buffer<bufferIndex>().upload_data(data);
		}
	}

	template <int bufferIndex>
	std::vector<BufferTypeFromIndex<bufferIndex>> download_buffer() const
	{
		if constexpr (IsCPUBuffer::value)
			return get_buffer<bufferIndex>();
		else
			return get_buffer<bufferIndex>().download_data();
	}

	template <int bufferIndex>
	auto& get_buffer()
	{
		return std::get<bufferIndex>(buffers);
	}

	template <int bufferIndex>
	const auto& get_buffer() const
	{
		return std::get<bufferIndex>(buffers);
	}

	template <int bufferIndex>
	auto* get_buffer_data_ptr()
	{
		return std::get<bufferIndex>(buffers).data();
	}

	template <int bufferIndex>
	auto* get_buffer_data_atomic_ptr()
	{
		if constexpr (IsCPUBuffer::value)
			return std::get<bufferIndex>(buffers).data();
		else
			// For the GPU, calling the 'get_atomic_device_pointer' of OrochiBuffer
			return std::get<bufferIndex>(buffers).get_atomic_device_pointer();
	}

	template <int bufferIndex>
	void upload_to_buffer(const std::vector<BufferTypeFromIndex<bufferIndex>>& data)
	{
		if constexpr (IsCPUBuffer::value)
		{
			// If our main container type for this SoA is std::vector (i.e. this is for the CPU), then we're uploading
			// to the buffer simply by copying
			get_buffer<bufferIndex>() = data;
		}
		else
		{
			// If our main container type for this SoA is OrochiBuffer (i.e. this is for the GPU), then we're uploading
			// to the buffer by uploading to the GPU
			get_buffer<bufferIndex>().upload_data(data);
		}
	}

	template <int bufferIndex>
	void upload_to_buffer_partial(size_t start_index, const std::vector<BufferTypeFromIndex<bufferIndex>>::const_iterator& iterator_start, size_t element_count)
	{
		if constexpr (IsCPUBuffer::value)
		{
			// If our main container type for this SoA is std::vector (i.e. this is for the CPU), then we're uploading
			// to the buffer simply by copying
			std::copy(iterator_start, iterator_start + element_count, get_buffer<bufferIndex>().begin() + start_index);
		}
		else
		{
			// If our main container type for this SoA is OrochiBuffer (i.e. this is for the GPU), then we're uploading
			// to the buffer by uploading to the GPU
			get_buffer<bufferIndex>().upload_data_partial(start_index, &*iterator_start, element_count);
		}
	}

	template <int bufferIndex>
	void upload_to_buffer_partial(size_t start_index, const BufferTypeFromIndex<bufferIndex>* iterator_start, size_t element_count)
	{
		if constexpr (IsCPUBuffer::value)
		{
			// If our main container type for this SoA is std::vector (i.e. this is for the CPU), then we're uploading
			// to the buffer simply by copying
			std::copy(iterator_start, iterator_start + element_count, get_buffer<bufferIndex>().begin() + start_index);
		}
		else
		{
			// If our main container type for this SoA is OrochiBuffer (i.e. this is for the GPU), then we're uploading
			// to the buffer by uploading to the GPU
			get_buffer<bufferIndex>().upload_data_partial(start_index, iterator_start, element_count);
		}
	}

	template <int bufferIndex>
	void upload_to_buffer_partial(size_t start_index, const std::vector<BufferTypeFromIndex<bufferIndex>>& data, size_t element_count)
	{
		upload_to_buffer_partial<bufferIndex>(start_index, data.data(), element_count);
	}

	void free()
	{
		m_maximum_size = 0;

		// Applies clear() on each buffer in the tuple
		std::apply(
			[](auto&... buffer)
			{
				if constexpr (IsCPUBuffer::value)
					// decltype here gives us the exact type of 'buffer' which can be std::vector<float>& for example,
					// **with** the reference type
					//
					// But we want to clear the buffer by overriding it with a newly instantiated buffer so we don't want
					// the reference, hence the use of std::decay_t
					((buffer = std::decay_t<decltype(buffer)>{}), ...);
				else
					((buffer.free()), ...);
			},
			buffers);
	}

private:
	template <std::size_t... indices>
	void resize_with_exclusions_internal(std::size_t new_element_count, const std::unordered_set<int>& excluded_buffer_indices, std::index_sequence<indices...>)
	{
		// If the current buffer being processed has an index that is excluded, let's not resize it
		((excluded_buffer_indices.count(indices) == 0 ? resize_buffer_internal(std::get<indices>(buffers), new_element_count) : void()), ...);
	}

	template <typename BufferType>
	void resize_buffer_internal(BufferType& buffer, std::size_t new_element_count)
	{
		buffer = std::decay_t<decltype(buffer)>(new_element_count);
	}

	std::tuple<Container<Types>...> buffers;

	std::size_t m_maximum_size = 0;
};

namespace GenericSoAHelpers
{
	template <template <typename> class BufferContainer, typename T, typename U>
	void memset_buffer(BufferContainer<T>& buffer, U memset_value)
	{
		if constexpr (std::is_same_v<BufferContainer<T>, std::vector<T>>)
		{
			// std::vector type

			if constexpr (IsStdAtomic<T>::value)
			{
				// For atomic types, we have to store into them with a loop because they do not have an =operator()
				// so we can't use std::fill
				for (auto& value : buffer)
					value.store(memset_value);
			}
			else
				std::fill(buffer.begin(), buffer.end(), memset_value);
		}
		else
		{
			std::vector<T> data(buffer.size(), memset_value);

			buffer.upload_data(data);
		}
	}

	template <template <typename> class BufferContainer, typename T>
	void resize(BufferContainer<T>& buffer, std::size_t new_size)
	{
		buffer = std::decay_t<decltype(buffer)>(new_size);
	}
} // namespace GenericSoAHelpers

#endif
