/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_PACKING_H
#define HOST_DEVICE_COMMON_PACKING_H

#include "HostDeviceCommon/Color.h"

/**
 * Packs 88 bools into a uchar
 */
struct UChar8BoolsPacked
{
	/**
	 * Returns the bool packed at bit 'index'. 0 is LSB.
	 * 
	 * 'index' is in [0, 7]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE bool get_bool() const
	{
		return packed & (1 << index);
	}

	/**
	 * Sets the bool at bit 'index' in the packed data. 0 is LSB.
	 * 
	 * 'index' is in [0, 7]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_bool(bool value)
	{
		// Clear the bit
		packed &= ~(1 << index);

		// Sets
		packed |= (value ? 1 : 0) << index;
	}

private:
	unsigned char packed = 0;
};

/**
 * Packs a ColorRGB32F into 3x8 bit = 24 bits (this isn't a loss of precision 
 * for colors that already were in SDR [0, 255]).
 * 
 * A float in range [0, 1] can be packed in the remaining 8 bits. 
 * This leaves us with a precision of 0.004 between values in [0, 1]. Which is probably 
 * more than enough. Who picks up the difference between a roughness of 0.5 and 0.504 anyways?
 */
struct ColorRGB24bFloat0_1Packed
{
	static constexpr float inv_255		 = (1.0f / (255 << 0));
	static constexpr float inv_255_shl_8 = (1.0f / (255 << 8));

	HIPRT_HOST_DEVICE ColorRGB32F get_color() const
	{
		float r = static_cast<float>(packed & 0x000000FF) * inv_255;
		float g = static_cast<float>(packed & 0x0000FF00) * inv_255_shl_8;
		float b = static_cast<float>((packed & 0x00FF0000) >> 16) * inv_255;

		return ColorRGB32F(r, g, b);
	}

	HIPRT_HOST_DEVICE float get_float() const
	{
		return static_cast<float>((packed & 0xFF000000) >> 24) * inv_255;
	}

	HIPRT_HOST_DEVICE void set_color(const ColorRGB32F& color)
	{
		// Clear 24 lower bits
		packed &= 0xFF000000;

		// Set
		packed |= static_cast<unsigned char>(color.r * 255.0f);
		packed |= static_cast<unsigned char>(color.g * 255.0f) << 8;
		packed |= static_cast<unsigned char>(color.b * 255.0f) << 16;
	}

	HIPRT_HOST_DEVICE void set_float(float float_in_0_1)
	{
		// Clear
		packed &= 0x00FFFFFF;

		// Set
		packed |= static_cast<unsigned char>(float_in_0_1 * 255.0f) << 24;
	}

private:
	unsigned int packed = 0;
};

/**
 * 4 floats in [0, 1] all packed into a 32 bit uint.
 * 
 * This gives 8 bits for each float in [0, 1] --> precision of 0.004
 */
struct Float4xPacked
{
	static constexpr float inv_255 = 0.00392156862745098039f;

	/**
	 * Returns the float at index 'index'
	 * 
	 * 'index' must be in [0, 3]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE float get_float() const
	{ 
		return static_cast<float>((packed & (0xFF << (index * 8))) >> (index * 8)) * inv_255;
	}

	/**
	 * Sets the float number 'index' of this 4x packed float
	 * 
	 * 'index' must be in [0, 3]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_float(float value)
	{
		// Clear
		packed &= ~(0xFF << (index * 8));

		// Set
		packed |= static_cast<unsigned char>(value * 255.0f) << (index * 8);
	}

private:
	unsigned int packed = 0;
};

/**
 * 2 floats in [0, 1] and 2 unsigned chars all packed into a 32 bit uint.
 *
 * This gives 8 bits for each float in [0, 1] --> precision of 0.004
 */
struct Float2xUChar2xPacked
{
	static constexpr float inv_255 = 0.00392156862745098039f;

	/**
	 * Returns one of the float packed in this structure
	 * 
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE float get_float() const
	{
		return ((packed & (0xFF << (index * 8))) >> (index * 8)) * inv_255;
	}

	/**
	 * Returns one of the unsigned char packed in this structure
	 *
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE unsigned char get_uchar() const
	{
		return (packed & (0x00FF0000 << (index * 8))) >> (index * 8 + 16);
	}
	
	/**
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_float(float value)
	{
		// Clear
		packed &= ~(0xFF << (index * 8));

		// Set
		packed |= static_cast<unsigned char>(value * 255.0f) << (index * 8);
	}

	/**
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_uchar(unsigned char value)
	{
		// Clear
		packed &= ~(0x00FF0000 << (index * 8));

		// Set
		packed |= value << (index * 8 + 16);
	}

private:
	// Floats are in the 16 LSB
	// Uchars are in the 16 MSB
	unsigned int packed = 0;
};

/**
 * Packs two uints 16bits into one 32bit
 */
struct Uint2xPacked
{
	/**
	 * Index must be 0 or 1
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE unsigned short get_value() const
	{
		return (packed & (0xFFFF << (index * 16))) >> (index * 16);
	}

	/**
	 * Index must be 0 or 1
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_value(unsigned short value)
	{
		// Clear
		packed &= ~(0xFFFF << (index * 16));

		// Set
		packed |= value << (index * 16);
	}

private:
	unsigned int packed = 0;
};

#endif
