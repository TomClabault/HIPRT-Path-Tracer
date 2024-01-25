#pragma once

#include <Orochi/OrochiUtils.h>
#include <utility>

namespace Oro
{

/// @brief A helper function that casts an address of a pointer to the device memory to a void pointer to be used as an argument for kernel calls. 
/// @tparam T The type of the element stored in the device memory.
/// @param ptr The address of a pointer to the device memory.
/// @return A void pointer.
template<typename T>
void* arg_cast( T* const* ptr ) noexcept
{
	return reinterpret_cast<void*>( const_cast<T**>( ptr ) );
}

template<typename T>
class GpuMemory final
{
  public:
	GpuMemory() = default;

	/// @brief Allocate the device memory with the given size.
	/// @param init_size The initial size which represents the number of elements.
	explicit GpuMemory( const size_t init_size )
	{
		OrochiUtils::malloc( m_data, init_size );

		m_size = init_size;
		m_capacity = init_size;
	}

	GpuMemory( const GpuMemory& ) = delete;
	GpuMemory& operator=( const GpuMemory& other ) = delete;

	GpuMemory( GpuMemory&& other ) noexcept : m_data{ std::exchange( other.m_data, nullptr ) }, m_size{ std::exchange( other.m_size, 0ULL ) }, m_capacity{ std::exchange( other.m_capacity, 0ULL ) } {}

	GpuMemory& operator=( GpuMemory&& other ) noexcept
	{
		GpuMemory tmp( std::move( *this ) );

		swap( *this, other );

		return *this;
	}

	~GpuMemory()
	{
		if( m_data )
		{
			OrochiUtils::free( m_data );
			m_data = nullptr;
		}
		m_size = 0ULL;
		m_capacity = 0ULL;
	}

	/// @brief  Get the size of the device memory.
	/// @return The size of the device memory.
	size_t size() const noexcept { return m_size; }

	/// @brief Get the pointer to the device memory.
	/// @return The pointer to the device memory.
	T* ptr() const noexcept { return m_data; }

	/// @brief Get the address of the pointer to the device memory. Useful for passing arguments to the kernel call.
	/// @return The address of the pointer to the device memory.
	T* const* address() const noexcept { return &m_data; }

	/// @brief Resize the device memory. Its capacity is unchanged if the new size is smaller than the current one.
	/// The old data should be considered invalid to be used after the function is called unless @c copy is set to True.
	/// @param new_size The new memory size after the function is called.
	/// @param copy If true, the function will copy the data to the newly created memory space as well.
	void resize( const size_t new_size, const bool copy = false ) noexcept
	{
		if( new_size <= m_capacity )
		{
			m_size = new_size;
			return;
		}

		GpuMemory tmp( new_size );

		if( copy )
		{
			OrochiUtils::copyDtoD( tmp.m_data, m_data, m_size );
		}

		*this = std::move( tmp );
	}

	/// @brief Reset the memory space so that all bits inside are cleared to zero.
	void reset() noexcept { OrochiUtils::memset( m_data, 0, m_size * sizeof( T ) ); }

	/// @brief Copy the data from device memory to host.
	/// @param host_ptr The host pointer.
	/// @param host_data_size The size of the host memory which represents the number of elements.
	void copyFromHost( const T* host_ptr, const size_t host_data_size ) noexcept
	{
		resize( host_data_size );
		OrochiUtils::copyHtoD( m_data, host_ptr, host_data_size );
	}

	/// @brief Get the content of the first element stored in the device memory.
	/// @return The content of the first element in the device memory.
	T getSingle() const noexcept
	{
		T result{};

		OrochiUtils::copyDtoH( &result, m_data, 1ULL );

		return result;
	}

	/// @brief Get all the data stored in the device memory.
	/// @return A vector which contains all the data stored in the device memory.
	std::vector<T> getData() const noexcept
	{
		std::vector<T> result{};
		result.resize( m_size );

		OrochiUtils::copyDtoH( result.data(), m_data, m_size );

		return result;
	}

  private:
	static void swap( GpuMemory& lhs, GpuMemory& rhs ) noexcept
	{
		std::swap( lhs.m_data, rhs.m_data );
		std::swap( lhs.m_size, rhs.m_size );
		std::swap( lhs.m_capacity, rhs.m_capacity );
	}

	T* m_data{ nullptr };
	size_t m_size{ 0ULL };
	size_t m_capacity{ 0ULL };
};

} // namespace Oro
