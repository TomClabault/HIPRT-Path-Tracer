/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef LTC_FITTER_H
#define LTC_FITTER_H

#include "HostDeviceCommon/Math.h"

#include <iostream>

/**
 * Implementation adapted from the code given with the paper
 * [Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, Heitz et al., 2016]
 */

struct LTC
{
	// lobe amplitude
	float amplitude;

	// parametric representation
	float m11, m22, m13, m23;
	float3 X, Y, Z;

	// matrix representation
	float3x3 M;
	float3x3 invM;
	float detM;

	LTC()
	{
		amplitude = 1;

		m11 = 1;
		m22 = 1;
		m13 = 0;
		m23 = 0;

		X = float3(1, 0, 0);
		Y = float3(0, 1, 0);
		Z = float3(0, 0, 1);

		update();
	}

	void copy(const LTC& ltc)
	{
		this->amplitude = ltc.amplitude;

		this->m11 = ltc.m11;
		this->m22 = ltc.m22;
		this->m13 = ltc.m13;
		this->m23 = ltc.m23;

		this->X = ltc.X;
		this->Y = ltc.Y;
		this->Z = ltc.Z;
		this->M = ltc.M;

		this->invM = ltc.invM;
		this->detM = ltc.detM;
	}

	void update() // compute matrix from parameters
	{
		M = float3x3(X, Y, Z) *
			float3x3(m11, 0, 0,
				0, m22, 0,
				m13, m23, 1);
		invM = inverse(M);
		detM = abs(determinant(M));
	}

	void update2()
	{
		invM = inverse(M);
		detM = hippt::abs(determinant(M));
	}

	float eval(const float3& L) const
	{
		float3 Loriginal = hippt::normalize(invM * L);
		float3 L_ = M * Loriginal;

		float l = hippt::length(L_);
		float Jacobian = detM / (l * l * l);

		float D = 1.0f / 3.14159f * hippt::max<float>(0.0f, Loriginal.z);

		float res = amplitude * D / Jacobian;
		return res;
	}

	float3 sample(const float U1, const float U2) const
	{
		const float theta = acosf(sqrtf(U1));
		const float phi = 2.0f * 3.14159f * U2;
		const float3 L = hippt::normalize(M * make_float3(sinf(theta) * cosf(phi), sinf(theta) * sinf(phi), cosf(theta)));
		return L;
	}

	void testNormalization() const
	{
		double sum = 0;
		float dtheta = 0.005f;
		float dphi = 0.005f;

		for (float theta = 0.0f; theta <= 3.14159f; theta += dtheta)
		{
			for (float phi = 0.0f; phi <= 2.0f * 3.14159f; phi += dphi)
			{
				float3 L(cosf(phi) * sinf(theta), sinf(phi) * sinf(theta), cosf(theta));

				sum += sinf(theta) * eval(L);
			}
		}

		sum *= dtheta * dphi;
		std::cout << "LTC normalization test: " << sum << std::endl;
		std::cout << "LTC normalization expected: " << amplitude << std::endl;
	}
};




/*
#include "ltc_alpha.inc"

 // build orthonormal basis (Building an Orthonormal Basis from a 3D Unit Vector Without Normalization, [Frisvad2012])
 void buildOrthonormalBasis(float3& omega_1, float3& omega_2, const float3& omega_3)
{
	if(omega_3.z < -0.9999999f)
	{
	   omega_1 = float3 ( 0.0f , -1.0f , 0.0f );
	   omega_2 = float3 ( -1.0f , 0.0f , 0.0f );
	} else {
	   const float a = 1.0f /(1.0f + omega_3.z );
	   const float b = -omega_3.x*omega_3 .y*a ;
	   omega_1 = float3 (1.0f - omega_3.x*omega_3. x*a , b , -omega_3.x );
	   omega_2 = float3 (b , 1.0f - omega_3.y*omega_3.y*a , -omega_3.y );
	}
}

float3x3 moment2M(const float3x3& Sigma, const float3& average)
{
	float3 T1, T2;
	buildOrthonormalBasis(T1, T2, average);

	const float var1 = dot(T1, Sigma * T1);
	const float var2 = dot(T2, Sigma * T2);
	const float c12 = dot(T1, Sigma * T2);

	mat2 Sigma12(var1, c12, c12, var2);
	vec2 eigen1(1,1);
	eigen1 = normalize(Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * eigen1);
	vec2 eigen2(-eigen1.y, eigen1.x);

	float3 Teigen1 = eigen1.x * T1 + eigen1.y * T2;
	float3 Teigen2 = eigen2.x * T1 + eigen2.y * T2;

	float sigma1 = sqrtf(dot(Teigen1, Sigma * Teigen1));
	float sigma2 = sqrtf(dot(Teigen2, Sigma * Teigen2));

	int index1 = std::min(size, std::max(0, (int)floorf(size * sigma1)));
	int index2 = std::min(size, std::max(0, (int)floorf(size * sigma2)));

	return float3x3(Teigen1, Teigen2, average) * float3x3(tabAlpha[index1+size*index2].x, 0, 0, 0, tabAlpha[index1+size*index2].y, 0, 0, 0, 1.0f);
}
*/

#endif



