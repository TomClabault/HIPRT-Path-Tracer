/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_GENERIC_SOA_H
#define RENDERER_GENERIC_SOA_H

#include <tuple>
#include <cstddef>
#include <utility>

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
    typename... Ts>
struct GenericSoA
{
    template <typename T>
    using BufferValueType = typename std::decay_t<T>::value_type;

    template <int bufferIndex>
    using BufferTypeFromIndex = typename std::tuple_element<bufferIndex, std::tuple<Container<Ts>...>>::type::value_type;

    std::tuple<Container<Ts>...> buffers;

    void resize(std::size_t new_element_count)
    {
        // Applies resize(new_element_count) on each buffer in the tuple
        std::apply([new_element_count](auto&... buffer) { (buffer.resize(new_element_count), ...); }, buffers);
    }

    std::size_t get_byte_size() const
    {
        std::size_t total = 0;

        // For each container, add sizeof(value_type) * size()
        std::apply([&](auto const&... buffer) 
        {
            ((total += buffer.size() * sizeof(BufferValueType<decltype(buffer)>)), ...);
        }, buffers);

        return total;
    }

	unsigned int size() const
	{
        return std::get<0>(buffers).size();
	}

    template<int bufferIndex>
    auto& get_buffer()
    {
        return std::get<bufferIndex>(buffers);
    }

    template <int bufferIndex>
    void upload_to_buffer(const std::vector<BufferTypeFromIndex<bufferIndex>>& data)
    {
        if constexpr (std::is_same_v<Container<BufferTypeFromIndex<bufferIndex>>, std::vector<BufferTypeFromIndex<bufferIndex>>>)
        {
            // If our main container type for this SoA is std::vector (i.e. this is for the CPU), then we're uploading
            // to the buffer simply by copying
            get_buffer<bufferIndex>() = data;
        }
        else if constexpr (std::is_same_v<Container<BufferTypeFromIndex<bufferIndex>>, OrochiBuffer<BufferTypeFromIndex<bufferIndex>>>)
        {
            // If our main container type for this SoA is OrochiBuffer (i.e. this is for the GPU), then we're uploading
            // to the buffer by uploading to the GPU
            get_buffer<bufferIndex>().upload_data(data);
        }
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
};

#endif
