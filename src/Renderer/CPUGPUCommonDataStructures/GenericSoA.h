/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GENERIC_SOA_H
#define RENDERER_GENERIC_SOA_H

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "HostDeviceCommon/AtomicType.h"

template<typename T, template<typename> class Container>
using GenericAtomicType = typename std::conditional_t<std::is_same<Container<T>, std::vector<T>>::value, AtomicType<T>, T>;

// Helper to detect std::atomic<...>
//
// std::false_type and std::true_type are structures that
// have ::value equal to 'false' or ::value equal to 'true' respectively
//
// By inheriting from std::false_type or std::true_type, we can check at compile time
// what's our ::value and use a constexpr if() on that
template<typename T>
struct IsStdAtomic : std::false_type {};

template<typename U>
struct IsStdAtomic<std::atomic<U>> : std::true_type {};

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
template<
    template<typename> class Container,
    typename... Types>
struct GenericSoA
{
    template <typename T>
    using BufferTypeFromVariable = typename std::decay_t<T>::value_type;

    template <int bufferIndex>
    using BufferTypeFromIndex = typename std::tuple_element<bufferIndex, std::tuple<Container<Types>...>>::type::value_type;

    using IsCPUBuffer = std::is_same<Container<BufferTypeFromIndex<0>>, std::vector<BufferTypeFromIndex<0>>>;

    void resize(std::size_t new_element_count, std::unordered_set<int> excluded_buffer_indices = {})
    {
        // Applies resize(new_element_count) on each buffer in the tuple and handles the excluded buffers
        resize_with_exclusions_internal(new_element_count, excluded_buffer_indices, std::index_sequence_for<Types...>{});
    }

    std::size_t get_byte_size() const
    {
        std::size_t total = 0;

        // For each container, add sizeof(value_type) * size()
        std::apply([&](auto const&... buffer) 
        {
            ((total += buffer.size() * sizeof(BufferTypeFromVariable<decltype(buffer)>)), ...);
        }, buffers);

        return total;
    }

	unsigned int size() const
	{
        return std::get<0>(buffers).size();
	}

    template<int bufferIndex>
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
            std::vector<BufferTypeFromIndex<bufferIndex>> data(size(), memset_value);
            get_buffer<bufferIndex>().upload_data(data);
        }
    }

    template<int bufferIndex>
    auto& get_buffer()
    {
        return std::get<bufferIndex>(buffers);
    }

    template<int bufferIndex>
    auto* get_buffer_data_ptr()
    {
        return std::get<bufferIndex>(buffers).data();
    }

    template<int bufferIndex>
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
    void upload_to_buffer_partial(size_t start_index, const std::vector<BufferTypeFromIndex<bufferIndex>>& data, size_t element_count)
    {
        upload_to_buffer_partial<bufferIndex>(start_index, data.begin(), element_count);
    }

    void free()
    {
        // Applies clear() on each buffer in the tuple
        std::apply([](auto&... buffer)
        {
            // decltype here gives us the exact type of 'buffer' which can be std::vector<float>& for example,
            // **with** the reference type
            //
            // But we want to clear the buffer by overriding it with a newly instantiated buffer so we don't want
            // the reference, hence the use of std::decay_t
            ((buffer = std::decay_t<decltype(buffer)>{}), ...);
        }, buffers);
    }

private:
    template <std::size_t... indices>
    void resize_with_exclusions_internal(std::size_t new_element_count, const std::unordered_set<int>& excluded_buffer_indices, std::index_sequence<indices...>)
    {
        // If the current buffer being processed has an index that is excluded, let's not resize it
        ((excluded_buffer_indices.count(indices) == 0
            ? resize_buffer_internal(std::get<indices>(buffers), new_element_count)
            : void()), ...);
    }

    template <typename BufferType>
    void resize_buffer_internal(BufferType& buffer, std::size_t new_element_count)
    {
        if constexpr (IsStdAtomic<typename BufferType::value_type>::value)
            // If the buffer is a buffer of std::atomic on the CPU, we cannot use resize
            // (because std::atomic are missing some operators used by
            // std::vector.resize() so we have to recreate the buffer instead
            buffer = std::decay_t<decltype(buffer)>(new_element_count);
        else
            buffer.resize(new_element_count);
    }

    std::tuple<Container<Types>...> buffers;
};

namespace GenericSoAHelpers
{
    template<template<typename> class BufferContainer, typename T, typename U>
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

    template<template<typename> class BufferContainer, typename T>
    void resize(BufferContainer<T>& buffer, std::size_t new_size)
    {
        if constexpr (IsStdAtomic<T>::value)
            // If the buffer is a buffer of std::atomic on the CPU, we cannot use resize
            // (because std::atomic are missing some operators used by
            // std::vector.resize() so we have to recreate the buffer instead
            buffer = std::decay_t<decltype(buffer)>(new_size);
        else
            buffer.resize(new_size);
    }
}

#endif
