/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_TYPE_TRAITS_H
#define HOST_DEVICE_COMMON_TYPE_TRAITS_H

/**
 * Type traits usable in the GPU kernels.
 *
 * The kernels can't include the headers of the C++ standard library (<type_traits>, <limits>, ...): the runtime compiler finds the headers of the
 * standard library of the installed MSVC, and recent versions of these headers refuse to compile with the Clang version of the runtime compiler
 */
namespace hippt
{
	template <typename A, typename B>
	struct is_same
	{
		static constexpr bool value = false;
	};

	template <typename A>
	struct is_same<A, A>
	{
		static constexpr bool value = true;
	};

	template <typename A, typename B>
	inline constexpr bool is_same_v = is_same<A, B>::value;

	/**
	 * Limits of the arithmetic types. As std::numeric_limits, max() and lowest() return T() for the types that have no specialization
	 */
	template <typename T>
	struct numeric_limits
	{
		static constexpr T max()
		{
			return T();
		}

		static constexpr T lowest()
		{
			return T();
		}
	};

	template <>
	struct numeric_limits<float>
	{
		static constexpr float max()
		{
			return 3.402823466e+38f;
		}

		static constexpr float lowest()
		{
			return -3.402823466e+38f;
		}
	};

	template <>
	struct numeric_limits<int>
	{
		static constexpr int max()
		{
			return 2147483647;
		}

		static constexpr int lowest()
		{
			return -2147483647 - 1;
		}
	};

	template <>
	struct numeric_limits<unsigned int>
	{
		static constexpr unsigned int max()
		{
			return 0xFFFFFFFFu;
		}

		static constexpr unsigned int lowest()
		{
			return 0u;
		}
	};
} // namespace hippt

#endif // #ifndef HOST_DEVICE_COMMON_TYPE_TRAITS_H
