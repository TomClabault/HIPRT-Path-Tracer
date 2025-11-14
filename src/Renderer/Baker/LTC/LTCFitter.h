/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef RENDERER_BAKER_LTC_FITTER_H
#define RENDERER_BAKER_LTC_FITTER_H

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Image/Image.h"
#include "HostDeviceCommon/Material/MaterialUnpacked.h"
#include "HostDeviceCommon/Maths/Math.h"
#include "HostDeviceCommon/RenderData.h"
#include "HostDeviceCommon/Xorshift.h"
#include "Renderer/Baker/LTC/NelderMead.h"
#include "Renderer/CPUGPUCommonDataStructures/BSDFDataHost.h"

#include <iostream>

 /**
  * Implementation adapted from the code given with the paper
  * [Real-Time Polygonal-Light Shading with Linearly Transformed Cosines, Heitz et al., 2016]
  */

// minimal roughness (avoid singularities)
const float MIN_ALPHA = 0.0001f;

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

struct LTCFit
{
	LTCFit(LTC& ltc_, 
		const HIPRTRenderData& render_data_, const DeviceUnpackedEffectiveMaterial& material_, std::shared_ptr<BSDFContext> base_bsdf_context_, Xorshift32Generator& rng_,
		bool isotropic_, const float3& V_, unsigned int error_samples_, float alpha_) :
		ltc(ltc_), 
		render_data(render_data_), material(material_), base_bsdf_context(base_bsdf_context_), rng(rng_),
		V(V_), error_samples(error_samples_), alpha(alpha_), isotropic(isotropic_) { }

	void update(const float* params);

	// compute the error between the BRDF and the LTC
	// using Multiple Importance Sampling
	float compute_error(const LTC& ltc, const float3& V, const float alpha);

	float operator()(const float* params);

	LTC& ltc;
	const DeviceUnpackedEffectiveMaterial& material;
	const HIPRTRenderData& render_data;
	std::shared_ptr<BSDFContext> base_bsdf_context;
	Xorshift32Generator& rng;

	unsigned int error_samples = 48;

	bool isotropic;

	float3 V;
	float alpha;
};

class LTCFitter
{
public:
	LTCFitter() {}
	LTCFitter(const DeviceUnpackedEffectiveMaterial& material_);

	// fit data
	void fit(int resolution, int error_samples);

	bool is_fitting_done() const { return fitting_done; }

	void export_fitted_data_float3x3_C(bool export_inverse = false, bool export_amplitude = false);
	void export_fitted_data_float4_C(bool export_inverse = false, bool export_amplitude = false);

private:
	float compute_norm(const float3& V, const float alpha, Xorshift32Generator& rng);

	// compute the average direction of the BRDF
	float3 compute_average_dir(const float3& V, const float alpha, Xorshift32Generator& rng);

	// fit brute force
	// refine first guess by exploring parameter space
	void fit_internal(LTC& ltc, Xorshift32Generator& rng, const float3& V, const float alpha, const float epsilon = 0.05f, const bool isotropic = false);
	
	DeviceUnpackedEffectiveMaterial material;

	HIPRTRenderData m_render_data;

	// All precomputed tabulated data for BSDFs (directional albedo, LTC fits, ...)
	BSDFDataHost m_bsdf_data_cpu_data;

	RayVolumeState m_ray_volume_state;
	BSDFIncidentLightInfo m_incident_light_info;
	std::shared_ptr<BSDFContext> m_base_bsdf_context;

	bool fitting_done = false;
	unsigned int m_fit_resolution = 0;
	unsigned int m_error_samples = 48;

	std::vector<float3x3> m_fitted_data;
	std::vector<float2> m_tab_amplitude;
};

#endif
