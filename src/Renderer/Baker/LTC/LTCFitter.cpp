// fitLTC.cpp : Defines the entry point for the console application.
//
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Device/includes/BSDFs/Principled.h"
#include "Renderer/Baker/LTC/LTCFitter.h"
#include "Renderer/CPURenderer.h"

void LTCFit::update(const float* params)
{
	float m11 = std::max<float>(params[0], MIN_ROUGHNESS);
	float m22 = std::max<float>(params[1], MIN_ROUGHNESS);
	float m13 = params[2];
	float m23 = params[3];

	if (isotropic)
	{
		ltc.m11 = m11;
		ltc.m22 = m11;
		ltc.m13 = 0.0f;
		ltc.m23 = 0.0f;
	}
	else
	{
		ltc.m11 = m11;
		ltc.m22 = m22;
		ltc.m13 = m13;
		ltc.m23 = m23;
	}

	ltc.update();
}

// compute the error between the BRDF and the LTC
// using Multiple Importance Sampling
float LTCFit::compute_error(const LTC& ltc, const float3& V, const float roughness)
{
	double error = 0.0;

	int valid_sample_count = 0;
	for (int j = 0; j < error_samples; ++j)
	{
		for (int i = 0; i < error_samples; ++i)
		{
			const float U1 = (i + 0.5f) / (float)error_samples;
			const float U2 = (j + 0.5f) / (float)error_samples;

			double sample_error = 0.0;

			// importance sample LTC
			{
				// sample
				const float3 L = ltc.sample(U1, U2);

				BSDFContext bsdf_context = *base_bsdf_context;
				bsdf_context.view_direction = V;
				bsdf_context.to_light_direction = L;
				bsdf_context.material.roughness = roughness;

				// error with MIS weight
				float pdf_brdf;
				float eval_brdf = principled_bsdf_eval(render_data, bsdf_context, pdf_brdf).luminance() * L.z;

				float eval_ltc = ltc.eval(L);
				float pdf_ltc = eval_ltc / ltc.amplitude;
				double error_ = fabsf(eval_brdf - eval_ltc);
				error_ = error_ * error_ * error_;
				sample_error += error_ / (pdf_ltc + pdf_brdf);
			}

			// importance sample BRDF
			{
				// Sample
				BSDFContext bsdf_context = *base_bsdf_context;
				bsdf_context.view_direction = V;
				bsdf_context.material.roughness = roughness;

				float pdf_brdf;
				float3 sampled_direction;
				float eval_brdf = principled_bsdf_sample(render_data, bsdf_context, sampled_direction, pdf_brdf, rng).luminance() * sampled_direction.z;
				if (pdf_brdf == 0.0f)
					// Bad sample
					continue;

				// error with MIS weight
				float eval_ltc = ltc.eval(sampled_direction);
				float pdf_ltc = eval_ltc / ltc.amplitude;
				double error_ = fabsf(eval_brdf - eval_ltc);
				error_ = error_ * error_ * error_;
				sample_error += error_ / (pdf_ltc + pdf_brdf);
			}

			error += sample_error;

			valid_sample_count++;
		}
	}

	return (float)error / (float)(valid_sample_count);
}

float LTCFit::operator()(const float* params)
{
	update(params);

	return compute_error(ltc, V, roughness);
}

LTCFitter::LTCFitter(const DeviceUnpackedEffectiveMaterial& material_) : material(material_)
{
	m_base_bsdf_context = std::make_shared<BSDFContext>(
		make_float3(0, 0, 1), // view direction, will be overwritten at fitting time
		make_float3(0, 0, 1), // shading normal, will be overwritten at fitting time
		make_float3(0, 0, 1), // geometric normal, will be overwritten at fitting time
		make_float3(0, 0, 1), // to light direction, will be overwritten at fitting time
		m_incident_light_info,
		m_ray_volume_state,
		false,
		material,
		0, 0.0f);

	m_bsdf_data_cpu_data.load_bsdf_data(m_render_data);

	int out_trash, out_trash2;
	bool inside_trash;
	m_ray_volume_state.interior_stack.push(out_trash, out_trash2, inside_trash, NestedDielectricsInteriorStack::MAX_MATERIAL_INDEX, StackPriorityEntry::PRIORITY_MAXIMUM);
}

void LTCFitter::fit(int resolution, int error_samples)
{
	fitting_done = false;

	m_bsdf_data_cpu_data.to_device(m_render_data);

	m_fitted_data.resize(resolution * resolution);
	m_tab_amplitude.resize(resolution * resolution);
	m_fit_resolution = resolution;
	m_error_samples = error_samples;

	// loop over theta and roughness
	for (int a = resolution - 1; a >= 0; a--)
	{
#pragma omp parallel for
		for (int t = 0; t < resolution; t++)
		{
			Xorshift32Generator thread_rng(a * resolution + t + 1);

			LTC ltc;

			float roughness = std::max(MIN_ROUGHNESS, a / float(resolution - 1));
			float theta = std::min<float>(1.57f, t / float(resolution - 1) * 1.57079f);
			const float3 V = float3(sinf(theta), 0, cosf(theta));

			ltc.amplitude = compute_norm(V, roughness, thread_rng);
			const float3 averageDir = compute_average_dir(V, roughness, thread_rng);
			bool isotropic;

			// 1. first guess for the fit
			// init the hemisphere in which the distribution is fitted
			// if theta == 0 the lobe is rotationally symmetric and aligned with Z = (0 0 1)
			if (t == 0)
			{
				ltc.X = float3(1, 0, 0);
				ltc.Y = float3(0, 1, 0);
				ltc.Z = float3(0, 0, 1);

				if (a == resolution - 1) // roughness = 1
				{
					ltc.m11 = 1.0f;
					ltc.m22 = 1.0f;
				}
				else // init with roughness of previous fit
				{
					ltc.m11 = std::max<float>(m_fitted_data[a + 1 + t * resolution].m[0][0], MIN_ROUGHNESS);
					ltc.m22 = std::max<float>(m_fitted_data[a + 1 + t * resolution].m[1][1], MIN_ROUGHNESS);
				}

				ltc.m13 = 0;
				ltc.m23 = 0;
				ltc.update();

				isotropic = true;
			}
			// otherwise use previous configuration as first guess
			else
			{
				float3 L = hippt::normalize(averageDir);
				float3 T1(L.z, 0, -L.x);
				float3 T2(0, 1, 0);
				ltc.X = T1;
				ltc.Y = T2;
				ltc.Z = L;

				ltc.update();

				isotropic = false;
			}

			// 2. fit (explore parameter space and refine first guess)
			float epsilon = 0.05f;
			fit_internal(ltc, thread_rng, V, roughness, epsilon, isotropic);

			// copy data
			m_fitted_data[a + t * resolution] = ltc.M;
			m_tab_amplitude[a + t * resolution].x = ltc.amplitude;
			m_tab_amplitude[a + t * resolution].y = 0;

			// kill useless coefs in matrix and normalize
			m_fitted_data[a + t * resolution].m[0][1] = 0;
			m_fitted_data[a + t * resolution].m[1][0] = 0;
			m_fitted_data[a + t * resolution].m[2][1] = 0;
			m_fitted_data[a + t * resolution].m[1][2] = 0;
			m_fitted_data[a + t * resolution] = 1.0f / m_fitted_data[a + t * resolution].m[2][2] * m_fitted_data[a + t * resolution];

			std::cout << "a = " << a << "\t t = " << t << std::endl;
			std::cout << "roughness = " << roughness << "\t theta = " << theta << std::endl;
			std::cout << std::endl;
			std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][0] << "\t " << m_fitted_data[a + t * resolution].m[1][0] << "\t " << m_fitted_data[a + t * resolution].m[2][0] << std::endl;
			std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][1] << "\t " << m_fitted_data[a + t * resolution].m[1][1] << "\t " << m_fitted_data[a + t * resolution].m[2][1] << std::endl;
			std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][2] << "\t " << m_fitted_data[a + t * resolution].m[1][2] << "\t " << m_fitted_data[a + t * resolution].m[2][2] << std::endl;
			std::cout << std::endl;
		}
	}

	fitting_done = true;
}

void LTCFitter::export_fitted_data_float3x3_C(bool export_inverse, bool export_amplitude)
{
	std::ofstream file("fitted_ltc_float3x3.h");

	file << std::fixed;
	file << std::setprecision(6);

	file << "static const std::array<float3x3, " << m_fit_resolution << " * " << m_fit_resolution << "> fitted_data = {" << std::endl;
	for (int t = 0; t < m_fit_resolution; ++t)
	{
		for (int a = 0; a < m_fit_resolution; ++a)
		{
			file << "float3x3(";
			file << m_fitted_data[a + t * m_fit_resolution].m[0][0] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[0][1] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[0][2] << "f, ";
			file << m_fitted_data[a + t * m_fit_resolution].m[1][0] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[1][1] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[1][2] << "f, ";
			file << m_fitted_data[a + t * m_fit_resolution].m[2][0] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[2][1] << "f, " << m_fitted_data[a + t * m_fit_resolution].m[2][2] << "f)";
			if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
				file << ", ";
			file << std::endl;
		}
	}
	file << "};" << std::endl << std::endl;

	if (export_inverse)
	{
		file << "static const std::array<float3x3, " << m_fit_resolution << " * " << m_fit_resolution << "> fitted_data_inverse = {" << std::endl;

		for (int t = 0; t < m_fit_resolution; ++t)
		{
			for (int a = 0; a < m_fit_resolution; ++a)
			{
				float3x3 Minv = inverse(m_fitted_data[a + t * m_fit_resolution]);

				file << "float3x3(";
				file << Minv.m[0][0] << "f, " << Minv.m[0][1] << "f, " << Minv.m[0][2] << "f, ";
				file << Minv.m[1][0] << "f, " << Minv.m[1][1] << "f, " << Minv.m[1][2] << "f, ";
				file << Minv.m[2][0] << "f, " << Minv.m[2][1] << "f, " << Minv.m[2][2] << "f)";
				if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
					file << ", ";
				file << std::endl;
			}
		}

		file << "};" << std::endl << std::endl;
	}

	if (export_amplitude)
	{
		file << "static const std::array<float, " << m_fit_resolution << " * " << m_fit_resolution << "> amplitude = {" << std::endl;

		for (int t = 0; t < m_fit_resolution; ++t)
		{
			for (int a = 0; a < m_fit_resolution; ++a)
			{
				file << m_tab_amplitude[a + t * m_fit_resolution].x << "f";
				if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
					file << ", ";
				file << std::endl;
			}
		}

		file << "};" << std::endl;
	}

	file.close();
}

void LTCFitter::export_fitted_data_float4_C(bool export_inverse, bool export_amplitude)
{
	std::ofstream file("fitted_ltc_float4.h");

	file << std::fixed;
	file << std::setprecision(6);

	file << "static const std::array<float4, " << m_fit_resolution << " * " << m_fit_resolution << "> ggx_specular_lambert_diffuse_ltc_fit_parameters = {" << std::endl;
	for (int t = 0; t < m_fit_resolution; ++t)
	{
		file << "\t/**\n\t * t = " << t << "\n\t */" << std::endl;

		for (int a = 0; a < m_fit_resolution; ++a)
		{
			float3x3 M = m_fitted_data[a + t * m_fit_resolution];

			if (a == 0 || a == m_fit_resolution / 2 || a == m_fit_resolution - 1)
				file << "\t// a = " << a << std::endl;
			file << "\tmake_float4(";
			file << M.m[0][0] << "f, " << M.m[0][2] << "f, ";
			file << M.m[1][1] << "f, ";
			file << M.m[2][0] << "f)";
			if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
				file << ", ";
			file << std::endl;
		}

		if (t < m_fit_resolution - 1)
			file << std::endl << std::endl << std::endl << std::endl << std::endl;
	}
	file << "};" << std::endl << std::endl;

	if (export_inverse)
	{
		file << "static const std::array<float4, " << m_fit_resolution << " * " << m_fit_resolution << "> ggx_specular_lambert_diffuse_ltc_inverse_fit_parameters = {" << std::endl;

		for (int t = 0; t < m_fit_resolution; ++t)
		{
			file << "\t/**\n\t * t = " << t << "\n\t */" << std::endl;

			for (int a = 0; a < m_fit_resolution; ++a)
			{
				float3x3 Minv = inverse(m_fitted_data[a + t * m_fit_resolution]);

				if (a == 0 || a == m_fit_resolution / 2 || a == m_fit_resolution - 1)
					file << "\t// a = " << a << std::endl;
				file << "\tmake_float4(";
				file << Minv.m[0][0] << "f, " << Minv.m[0][2] << "f, ";
				file << Minv.m[1][1] << "f, ";
				file << Minv.m[2][0] << "f)";
				if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
					file << ", ";
				file << std::endl;
			}

			if (t < m_fit_resolution - 1)
				file << std::endl << std::endl << std::endl << std::endl << std::endl;
		}

		file << "};" << std::endl << std::endl;
	}

	if (export_amplitude)
	{
		file << "static const std::array<float, " << m_fit_resolution << " * " << m_fit_resolution << "> amplitude = {" << std::endl;

		for (int t = 0; t < m_fit_resolution; ++t)
		{
			for (int a = 0; a < m_fit_resolution; ++a)
			{
				file << m_tab_amplitude[a + t * m_fit_resolution].x << "f";
				if (a != m_fit_resolution - 1 || t != m_fit_resolution - 1)
					file << ", ";
				file << std::endl;
			}
		}

		file << "};" << std::endl;
	}

	file.close();
}

float LTCFitter::compute_norm(const float3& V, const float roughness, Xorshift32Generator& rng)
{
	float norm = 0.0;

	for (int j = 0; j < m_error_samples; ++j)
	{
		for (int i = 0; i < m_error_samples; ++i)
		{
			// Sample
			BSDFContext bsdf_context = *m_base_bsdf_context;
			bsdf_context.view_direction = V;
			bsdf_context.material.roughness = roughness;

			float pdf;
			float3 sampled_direction;
			float eval = principled_bsdf_sample(m_render_data, bsdf_context, sampled_direction, pdf, rng).luminance() * sampled_direction.z;

			// accumulate
			norm += (pdf > 0) ? eval / pdf : 0.0f;
		}
	}

	return norm / (float)(m_error_samples * m_error_samples);
}

float3 LTCFitter::compute_average_dir(const float3& V, const float roughness, Xorshift32Generator& rng)
{
	float3 averageDir = float3(0, 0, 0);

	for (int j = 0; j < m_error_samples; ++j)
	{
		for (int i = 0; i < m_error_samples; ++i)
		{
			const float U1 = (i + 0.5f) / (float)m_error_samples;
			const float U2 = (j + 0.5f) / (float)m_error_samples;

			// Sample
			BSDFContext bsdf_context = *m_base_bsdf_context;
			bsdf_context.view_direction = V;
			bsdf_context.material.roughness = roughness;

			float pdf;
			float3 sampled_direction;
			float eval = principled_bsdf_sample(m_render_data, bsdf_context, sampled_direction, pdf, rng).luminance() * sampled_direction.z;

			// accumulate
			averageDir += (pdf > 0) ? eval / pdf * sampled_direction : float3(0, 0, 0);
		}
	}

	// clear y component, which should be zero with isotropic BRDFs
	averageDir.y = 0.0f;

	return hippt::normalize(averageDir);
}

void LTCFitter::fit_internal(LTC& ltc, Xorshift32Generator& rng, const float3& V, const float roughness, const float epsilon, const bool isotropic)
{
	float startFit[4] = { ltc.m11, ltc.m22, ltc.m13, ltc.m23 };
	float resultFit[4];

	LTCFit fitter(ltc,
		m_render_data, material, m_base_bsdf_context, rng,
		isotropic, V, m_error_samples, roughness);

	// Find best-fit LTC lobe (scale, alphax, alphay)
	float error = NelderMead<4>(resultFit, startFit, epsilon, 1e-5f, 100, fitter);

	// Update LTC with best fitting values
	fitter.update(resultFit);
}
