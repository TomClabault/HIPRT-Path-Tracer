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
		return m_packed & (1 << index);
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
		m_packed &= ~(1 << index);

		// Sets
		m_packed |= (value ? 1 : 0) << index;
	}

private:
	unsigned char m_packed = 0;
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
		float r = static_cast<float>(m_packed & 0x000000FF) * inv_255;
		float g = static_cast<float>(m_packed & 0x0000FF00) * inv_255_shl_8;
		float b = static_cast<float>((m_packed & 0x00FF0000) >> 16) * inv_255;

		return ColorRGB32F(r, g, b);
	}

	HIPRT_HOST_DEVICE float get_float() const
	{
		return static_cast<float>((m_packed & 0xFF000000) >> 24) * inv_255;
	}

	HIPRT_HOST_DEVICE void set_color(const ColorRGB32F& color)
	{
		// Clear 24 lower bits
		m_packed &= 0xFF000000;

		// Set
		m_packed |= static_cast<unsigned char>(color.r * 255.0f);
		m_packed |= static_cast<unsigned char>(color.g * 255.0f) << 8;
		m_packed |= static_cast<unsigned char>(color.b * 255.0f) << 16;
	}

	HIPRT_HOST_DEVICE void set_float(float float_in_0_1)
	{
		// Clear
		m_packed &= 0x00FFFFFF;

		// Set
		m_packed |= static_cast<unsigned char>(float_in_0_1 * 255.0f) << 24;
	}

private:
	unsigned int m_packed = 0;
};

/**
 * 4 floats in [0, 1] all packed into a 32 bit unsigned int.
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
		return static_cast<float>((m_packed & (0xFFu << (index * 8))) >> (index * 8)) * inv_255;
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
		m_packed &= ~(0xFFu << (index * 8));

		// Set
		m_packed |= static_cast<unsigned char>(value * 255.0f) << (index * 8);
	}

private:
	unsigned int m_packed = 0;
};

/**
 * 2 floats in [0, 1] and 2 unsigned chars all packed into a 32 bit unsigned int.
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
		return ((m_packed & (0xFF << (index * 8))) >> (index * 8)) * inv_255;
	}

	/**
	 * Returns one of the unsigned char packed in this structure
	 *
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE unsigned char get_uchar() const
	{
		return (m_packed & (0x00FF0000u << (index * 8))) >> (index * 8 + 16);
	}
	
	/**
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_float(float value)
	{
		// Clear
		m_packed &= ~(0xFFu << (index * 8));

		// Set
		m_packed |= static_cast<unsigned char>(value * 255.0f) << (index * 8);
	}

	/**
	 * 'index' must be in [0, 1]
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_uchar(unsigned char value)
	{
		// Clear
		m_packed &= ~(0x00FF0000u << (index * 8));

		// Set
		m_packed |= value << (index * 8 + 16);
	}

private:
	// Floats are in the 16 LSB
	// Uchars are in the 16 MSB
	unsigned int m_packed = 0;
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
		return (m_packed & (0xFFFFu << (index * 16))) >> (index * 16);
	}

	/**
	 * Index must be 0 or 1
	 */
	template <unsigned char index>
	HIPRT_HOST_DEVICE void set_value(unsigned short value)
	{
		// Clear
		m_packed &= ~(0xFFFFu << (index * 16));

		// Set
		m_packed |= value << (index * 16);
	}

private:
	unsigned int m_packed = 0;
};

/**
 * Reference:
 * 
 * [1] [Survey of Efficient Representations for Independent Unit Vectors, Cigolle et al., 2014]
 */
struct Octahedral24BitNormal
{
public:
	HIPRT_HOST_DEVICE void pack(float3 normal)
	{
		float2_to_Snorm12_2x_as_3UChar(octahedral_encode(normal), packed_x, packed_y, packed_z);
	}

	/**
	 * Returns the normal that was packed in there
	 * 
	 * The returned normal is normalized
	 */
	HIPRT_HOST_DEVICE float3 unpack()
	{
		float2 v = Snorm12_2x_as_UChar_to_float2(packed_x, packed_y, packed_z);
		return final_decode(v.x, v.y);
	}

private:
	HIPRT_HOST_DEVICE float pack_Snorm12_float(float f)
	{
		return roundf(hippt::clamp(0.0f, 2.0f, f + 1.0f) * 2047.0f);
	}

	HIPRT_HOST_DEVICE void Snorm12_2x_as_3Uchar(float2 s, unsigned char& out_x, unsigned char& out_y, unsigned char& out_z)
	{
		float3 u;
		u.x = s.x / 16.0f;
		float t = floorf(s.y / 256.0f);
		u.y = ((u.x - floorf(u.x)) * 256.0f) + t;
		u.z = s.y - (t * 256.0f);

		out_x = u.x;
		out_y = u.y;
		out_z = u.z;
	}

	HIPRT_HOST_DEVICE void float2_to_Snorm12_2x_as_3UChar(float2 v, unsigned char& out_x, unsigned char& out_y, unsigned char& out_z)
	{
		float2 s = make_float2(pack_Snorm12_float(v.x), pack_Snorm12_float(v.y));

		Snorm12_2x_as_3Uchar(s, out_x, out_y, out_z);
	}

	HIPRT_HOST_DEVICE float2 octahedral_encode(float3 v)
	{
		float l1norm_inv = 1.0f / (abs(v.x) + abs(v.y) + abs(v.z));
		float2 result = make_float2(v.x * l1norm_inv, v.y * l1norm_inv);
		if (v.z < 0.0f)
			result = (make_float2(1.0f) - make_float2(hippt::abs(result.y), hippt::abs(result.x))) * sign_not_zero(make_float2(result.x, result.y));

		return result;
	}

	HIPRT_HOST_DEVICE float sign_not_zero(float k)
	{
		return k >= 0.0f ? 1.0f : -1.0f;
	}

	HIPRT_HOST_DEVICE float2 sign_not_zero(float2 v)
	{
		return make_float2(sign_not_zero(v.x), sign_not_zero(v.y));
	}

	HIPRT_HOST_DEVICE float3 final_decode(float x, float y) 
	{
		float3 v = make_float3(x, y, 1.0f - abs(x) - abs(y));
		if (v.z < 0.0f) 
		{
			float2 temp = make_float2(v.x, v.y);
			v.x = (1.0f - hippt::abs(temp.y)) * sign_not_zero(temp.x);
			v.y = (1.0f - hippt::abs(temp.x)) * sign_not_zero(temp.y);
		}
		return hippt::normalize(v);
	}

	HIPRT_HOST_DEVICE float2 Snorm12_2x_as_Uchar_to_packed_float2(unsigned char x, unsigned char y, unsigned char z)
	{
		float2 s;

		float temp = y / 16.0f;
		s.x = x * 16.0f + floorf(temp);
		s.y = (temp - floorf(temp)) * 256.0f * 16.0f + z;

		return s;
	}

	HIPRT_HOST_DEVICE float unpack_Snorm12(float f)
	{
		return hippt::clamp(-1.0f, 1.0f, (f / 2047.0f) - 1.0f);
	}

	HIPRT_HOST_DEVICE float2 Snorm12_2x_as_UChar_to_float2(unsigned char x, unsigned char y, unsigned char z)
	{
		float2 s = Snorm12_2x_as_Uchar_to_packed_float2(x, y, z);
		return make_float2(unpack_Snorm12(s.x), unpack_Snorm12(s.y));
	}

	unsigned char packed_x;
	unsigned char packed_y;
	unsigned char packed_z;
};

/**
 * Reference: https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_RGBE.hlsli
 */
struct RGBE9995Packed
{
	// RGBE, aka R9G9B9E5_SHAREDEXP, is an unsigned float HDR pixel format where red, green,
	// and blue all share the same exponent.  The color channels store a 9-bit value ranging
	// from [0/512, 511/512] which multiplies by 2^Exp and Exp ranges from [-15, 16].
	// Floating point specials are not encoded.
	HIPRT_HOST_DEVICE void pack(ColorRGB32F rgb)
	{
		// To determine the shared exponent, we must clamp the channels to an expressible range
		const float kMaxVal = hippt::asfloat(0x477F8000); // 1.FF x 2^+15
		const float kMinVal = hippt::asfloat(0x37800000); // 1.00 x 2^-16

		// Non-negative and <= kMaxVal
		rgb.clamp(0.0f, kMaxVal);

		// From the maximum channel we will determine the exponent.  We clamp to a min value
		// so that the exponent is within the valid 5-bit range.
		float MaxChannel = hippt::max(hippt::max(kMinVal, rgb.r), hippt::max(rgb.g, rgb.b));

		// 'Bias' has to have the biggest exponent plus 15 (and nothing in the mantissa).  When
		// added to the three channels, it shifts the explicit '1' and the 8 most significant
		// mantissa bits into the low 9 bits.  IEEE rules of float addition will round rather
		// than truncate the discarded bits.  Channels with smaller natural exponents will be
		// shifted further to the right (discarding more bits).
		float Bias = hippt::asfloat((hippt::asuint(MaxChannel) + 0x07804000) & 0x7F800000);

		// Shift bits into the right places
		unsigned int R, G, B;
		R = hippt::asuint(rgb.r + Bias);
		G = hippt::asuint(rgb.g + Bias);
		B = hippt::asuint(rgb.b + Bias);

		unsigned int E = (hippt::asuint(Bias) << 4) + 0x10000000;
		m_packed = E | B << 18 | G << 9 | (R & 0x1FF);
	}

	HIPRT_HOST_DEVICE ColorRGB32F unpack() const
	{
		float3 rgb = make_float3(m_packed & 0x1FF, (m_packed >> 9) & 0x1FF, (m_packed >> 18) & 0x1FF);
		return ColorRGB32F(hippt::ldexp(rgb, static_cast<int>(m_packed >> 27) - 24));
	}

private:
	unsigned int m_packed;
};

#endif
