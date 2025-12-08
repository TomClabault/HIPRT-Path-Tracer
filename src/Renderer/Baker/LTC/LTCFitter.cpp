// fitLTC.cpp : Defines the entry point for the console application.
//
#include <algorithm>
#include <fstream>
#include <iomanip>

#include "Device/includes/BSDFs/BSDFContext.h"
#include "Device/includes/BSDFs/Principled.h"
#include "Renderer/Baker/LTC/LTCFitter.h"
#include "Renderer/Baker/LTC/nelder-mead.h"
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

std::vector<float3> m_cached_sampled_BSDF_directions;
std::vector<float> m_cached_sampled_BSDF_directions_PDF;
std::vector<float> m_cached_sampled_BSDF_directions_eval;

// compute the error between the BRDF and the LTC
// using Multiple Importance Sampling
float LTCFit::compute_error(const LTC& ltc, const float3& V, const float roughness, double* out_min_error_factor, double* out_max_error_factor)
{
	double error = 0.0;
	double min_error_factor = 1e20;
	double max_error_factor = -1e20;

	Xorshift32Generator local_rng(0xdeadbeef);

	int valid_sample_count = 0;
	for (int j = 0; j < error_samples; ++j)
	{
		for (int i = 0; i < error_samples; ++i)
		{
			double sample_error = 0.0;

			// importance sample LTC
			{
				// sample
				float U1 = local_rng();
				float U2 = local_rng();
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

				double ratio = (double)hippt::max(eval_brdf, eval_ltc) / (double)hippt::min(eval_brdf, eval_ltc);
				if (std::isfinite(ratio))
				{
					min_error_factor = hippt::min(min_error_factor, ratio);
					max_error_factor = hippt::max(max_error_factor, ratio);
				}
			}

			// importance sample BRDF
			{
				// Sample
				BSDFContext bsdf_context = *base_bsdf_context;
				bsdf_context.view_direction = V;
				bsdf_context.material.roughness = roughness;

				float pdf_brdf;
				float3 sampled_direction;
				float eval_brdf;
				if (m_roughness_index > -1 && m_theta_index > -1)
				{
					size_t index = m_roughness_index * m_fit_resolution * error_samples * error_samples 
						+ m_theta_index * error_samples * error_samples
						+ j * error_samples
						+ i;

					sampled_direction = m_cached_sampled_BSDF_directions[index];
					pdf_brdf = m_cached_sampled_BSDF_directions_PDF[index];
					eval_brdf = m_cached_sampled_BSDF_directions_eval[index];
				}
				else
				{
					// No caching
					eval_brdf = principled_bsdf_sample(render_data, bsdf_context, sampled_direction, pdf_brdf, rng).luminance();
					eval_brdf *= sampled_direction.z;
				}

				if (pdf_brdf == 0.0f)
					// Bad sample
					continue;
				
				// error with MIS weight
				float eval_ltc = ltc.eval(sampled_direction);
				float pdf_ltc = eval_ltc / ltc.amplitude;
				double error_ = fabsf(eval_brdf - eval_ltc);

				error_ = error_ * error_ * error_;
				sample_error += error_ / (pdf_ltc + pdf_brdf);

				double ratio = (double)hippt::max(eval_brdf, eval_ltc) / (double)hippt::min(eval_brdf, eval_ltc);
				if (std::isfinite(ratio))
				{
					min_error_factor = hippt::min(min_error_factor, ratio);
					max_error_factor = hippt::max(max_error_factor, ratio);
				}
			}

			error += sample_error;

			valid_sample_count++;
		}
	}

	if (out_min_error_factor)
		*out_min_error_factor = (float)min_error_factor;
	if (out_max_error_factor)
		*out_max_error_factor = (float)max_error_factor;

	return (float)error / (float)(valid_sample_count);
}

float LTCFit::operator()(const float* params)
{
	update(params);

	return compute_error(ltc, V, roughness);
}

float LTCFit::operator()(const std::vector<float>& params_vec)
{
	const float params[4] = { params_vec[0], params_vec[1], params_vec[2], params_vec[3] };

	update(params);

	return compute_error(ltc, V, roughness);
}

LTCFitter::LTCFitter(const DeviceUnpackedEffectiveMaterial& material_) : material(material_)
{
	m_base_bsdf_context = std::make_shared<BSDFContext>(
		make_float3(0, 0, 1), // view direction, will be overwritten at fitting time
		make_float3(0, 0, 1), // shading normal, canonical shading frame for LTC fitting
		make_float3(0, 0, 1), // geometric normal, canonical shading frame for LTC fitting
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

	m_cached_sampled_BSDF_directions.resize(resolution * resolution * error_samples * error_samples);
	m_cached_sampled_BSDF_directions_eval.resize(resolution * resolution * error_samples * error_samples);
	m_cached_sampled_BSDF_directions_PDF.resize(resolution * resolution * error_samples * error_samples);
	for (int a = resolution - 1; a >= 0; a--)
	{
#pragma omp parallel for
		for (int t = 0; t < resolution; t++)
		{
			Xorshift32Generator cache_rng(0xdeadbeef);

			float roughness = std::max(MIN_ROUGHNESS, a / float(resolution - 1));
			float theta = std::min<float>(1.57f, t / float(resolution - 1) * 1.57079f);
			const float3 V = float3(sinf(theta), 0, cosf(theta));

			BSDFContext bsdf_context = *m_base_bsdf_context;
			bsdf_context.view_direction = V;
			bsdf_context.material.roughness = roughness;

			for (int j = 0; j < error_samples; j++)
			{
				for (int i = 0; i < error_samples; i++)
				{
					size_t cache_index = a * resolution * error_samples * error_samples
						+ t * error_samples * error_samples
						+ j * error_samples
						+ i;

					m_cached_sampled_BSDF_directions_eval[cache_index] = principled_bsdf_sample(
						m_render_data,
						bsdf_context,
						m_cached_sampled_BSDF_directions[cache_index],
						m_cached_sampled_BSDF_directions_PDF[cache_index],
						cache_rng).luminance();
					m_cached_sampled_BSDF_directions_eval[cache_index] *= m_cached_sampled_BSDF_directions[cache_index].z;
				}
			}
		}
	}

	LTC ltc;
	// loop over theta and roughness
	for (int a = resolution - 1; a >= 0; a--)
	{
		for (int t = 0; t < resolution; t++)
		{
			Xorshift32Generator thread_rng(a * resolution + t + 1);

			float roughness = std::max(MIN_ROUGHNESS, a / float(resolution - 1));
			float theta = std::min<float>(1.57f, t / float(resolution - 1) * 1.57079f);
			const float3 V = float3(sinf(theta), 0, cosf(theta));

			ltc.amplitude = compute_norm(V, roughness, thread_rng);
			const glm::vec3 averageDir = compute_average_dir(V, roughness, thread_rng);
			bool isotropic;

			// 1. first guess for the fit
			// init the hemisphere in which the distribution is fitted
			// if theta == 0 the lobe is rotationally symmetric and aligned with Z = (0 0 1)
			if (t == 0)
			{
				ltc.X = glm::vec3(1, 0, 0);
				ltc.Y = glm::vec3(0, 1, 0);
				ltc.Z = glm::vec3(0, 0, 1);

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
				glm::vec3 L = glm::normalize(averageDir);
				glm::vec3 T1 = glm::vec3(L.z, 0, -L.x);
				glm::vec3 T2 = glm::vec3(0, 1, 0);
				ltc.X = T1;
				ltc.Y = T2;
				ltc.Z = L;

				ltc.update();

				isotropic = false;
			}

			// 2. fit (explore parameter space and refine first guess)
			float epsilon = 0.05f;
			fit_internal(ltc, thread_rng, V, roughness, epsilon, isotropic, a, t);

			// copy data
			m_fitted_data[a + t * resolution] = float3x3::from_cols(
				make_float3(ltc.M[0].r, ltc.M[0].g, ltc.M[0].b),
				make_float3(ltc.M[1].r, ltc.M[1].g, ltc.M[1].b),
				make_float3(ltc.M[2].r, ltc.M[2].g, ltc.M[2].b)
			);
			m_tab_amplitude[a + t * resolution].x = ltc.amplitude;
			m_tab_amplitude[a + t * resolution].y = 0;

			// kill useless coefs in matrix and normalize
			m_fitted_data[a + t * resolution].m[0][1] = 0;
			m_fitted_data[a + t * resolution].m[1][0] = 0;
			m_fitted_data[a + t * resolution].m[2][1] = 0;
			m_fitted_data[a + t * resolution].m[1][2] = 0;
			m_fitted_data[a + t * resolution] = 1.0f / m_fitted_data[a + t * resolution].m[2][2] * m_fitted_data[a + t * resolution];

			std::cout << "a = " << a << "\t t = " << t << "; roughness = " << roughness << "\t theta = " << theta << std::endl;
			/*std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][0] << "\t " << m_fitted_data[a + t * resolution].m[1][0] << "\t " << m_fitted_data[a + t * resolution].m[2][0] << std::endl;
			std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][1] << "\t " << m_fitted_data[a + t * resolution].m[1][1] << "\t " << m_fitted_data[a + t * resolution].m[2][1] << std::endl;
			std::cout << "\t" << m_fitted_data[a + t * resolution].m[0][2] << "\t " << m_fitted_data[a + t * resolution].m[1][2] << "\t " << m_fitted_data[a + t * resolution].m[2][2] << std::endl;*/
			std::cout << std::endl;
		}
	}

	fitting_done = true;
}

void LTCFitter::export_fitted_data_float3x3_C(const std::string& filename, bool export_inverse, bool export_amplitude)
{
	std::ofstream file(filename);

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

void LTCFitter::export_fitted_data_float4_C(const std::string& filename, bool export_inverse, bool export_amplitude)
{
	std::ofstream file(filename);

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

extern bool DEBUG_ON;

void LTCFitter::compute_fitted_error()
{
	double error_sum = 0.0;

	for (int roughness = 0; roughness < m_fit_resolution; roughness++)
	{
		for (int theta = 0; theta < m_fit_resolution; theta++)
		{
			float r = std::max(MIN_ROUGHNESS, roughness / float(m_fit_resolution - 1));
			float th = std::min<float>(1.57f, theta / float(m_fit_resolution - 1) * hippt::M_PI_TWO);

			if (roughness == 22 && theta == 16)
			{
				r = 0.698039f;
				th = acos(0.748451f);
			}

			float3 V = make_float3(sinf(th), 0, cosf(th));

/**
View dir shading space: make_float3(0.66319, 0, 0.748451)
To vertex A: make_float3(-0.518554, 0.829912, -0.205787)
To vertex B: make_float3(0.504647, -0.0877765, -0.858852)
To vertex C: make_float3(0.890518, 0.363779, 0.273209)
Bsdf at A: 0, B: 0, C: 0.0993495
Roughness: 0.698039

View dir: 0, 0.66319, 0.748451
Light dir: -0.363779, 0.890518, 0.273209
Half vec: -0.191992, 0.82, 0.539202
 */

			LTC ltc;
			ltc.amplitude = m_tab_amplitude[roughness + theta * m_fit_resolution].x;
			ltc.M = glm::mat3(
				glm::vec3(m_fitted_data[roughness + theta * m_fit_resolution].m[0][0], m_fitted_data[roughness + theta * m_fit_resolution].m[1][0], m_fitted_data[roughness + theta * m_fit_resolution].m[2][0]),
				glm::vec3(m_fitted_data[roughness + theta * m_fit_resolution].m[0][1], m_fitted_data[roughness + theta * m_fit_resolution].m[1][1], m_fitted_data[roughness + theta * m_fit_resolution].m[2][1]),
				glm::vec3(m_fitted_data[roughness + theta * m_fit_resolution].m[0][2], m_fitted_data[roughness + theta * m_fit_resolution].m[1][2], m_fitted_data[roughness + theta * m_fit_resolution].m[2][2])
			);
			ltc.invM = glm::inverse(ltc.M);
			ltc.detM = abs(glm::determinant(ltc.M));

			if (roughness == 22 && theta == 16)
			{
				float3 T, B;
				build_ONB_XZ_plane(make_float3(0, 0, 1), T, B, V);
				float3 L = hippt::normalize(make_float3(0.890518, 0.363779, 0.273209));
				std::cout << "View dir: " << V.x << ", " << V.y << ", " << V.z << std::endl;
				std::cout << "Local to light dir: " << L.x << ", " << L.y << ", " << L.z << std::endl;
				std::cout << "LTC Eval: " << ltc.eval(L) << std::endl;

				BSDFContext bsdf_context = *m_base_bsdf_context;
				bsdf_context.view_direction = V;
				bsdf_context.to_light_direction = L;
				bsdf_context.material.roughness = r;

				// error with MIS weight
				float pdf_brdf;

				DEBUG_ON = true;
				float eval_brdf = principled_bsdf_eval(m_render_data, bsdf_context, pdf_brdf).luminance() * L.z;

				std::cout << "Eval BRDF: " << eval_brdf << std::endl;
			}

			Xorshift32Generator rng(roughness * m_fit_resolution + theta + 1);

			double min_error, max_error;
			double error = LTCFit(ltc,
				m_render_data, material, m_base_bsdf_context,
				rng,
				false, V, m_error_samples, r, -1, -1, -1).compute_error(ltc, V, r, &min_error, &max_error);

			if (roughness == 22 && theta == 16)
			{
				std::cout << "Min error: " << min_error << "\t Max error: " << max_error << std::endl;
				std::cout << std::endl;
			}

			std::cout << roughness << "/" << theta << " - " << "Roughness: " << r << "\t Theta: " << th << "(" << (float)theta / m_fit_resolution << ")" "\t Error: " << error << std::endl;

			error_sum += error;
		}
	}

	std::cout << "Error sum: " << error_sum << std::endl;
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
			float eval = principled_bsdf_sample(m_render_data, bsdf_context, sampled_direction, pdf, rng).luminance();
			eval *= sampled_direction.z;

			// accumulate
			norm += (pdf > 0) ? eval / pdf : 0.0f;
		}
	}

	return norm / (float)(m_error_samples * m_error_samples);
}

glm::vec3 LTCFitter::compute_average_dir(const float3& V, const float roughness, Xorshift32Generator& rng)
{
	glm::vec3 averageDir = glm::vec3(0, 0, 0);

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
			float eval = principled_bsdf_sample(m_render_data, bsdf_context, sampled_direction, pdf, rng).luminance();
			eval *= sampled_direction.z;

			// accumulate
			averageDir += (pdf > 0) ? eval / pdf * glm::vec3(sampled_direction.x, sampled_direction.y, sampled_direction.z) : glm::vec3(0, 0, 0);
		}
	}

	// clear y component, which should be zero with isotropic BRDFs
	averageDir.y = 0.0f;

	return glm::normalize(averageDir);
}

void LTCFitter::fit_internal(LTC& ltc, Xorshift32Generator& rng, const float3& V, const float roughness, const float epsilon, const bool isotropic, int roughness_index, int theta_index)
{
	float startFit[4] = { ltc.m11, ltc.m22, ltc.m13, ltc.m23 };
	float resultFit[4];

	LTCFit fitter(ltc,
		m_render_data, material, m_base_bsdf_context, rng,
		isotropic, V, m_error_samples, roughness, roughness_index, theta_index, m_fit_resolution);

	// Find best-fit LTC lobe (scale, alphax, alphay)
	float error = NelderMead<4>(resultFit, startFit, epsilon, 1e-5f, 100, fitter);

	/*std::vector<float> startFitVec = { startFit[0], startFit[1], startFit[2], startFit[3] };
	std::vector<float> resultFitVec = nelder_mead::find_min(fitter, startFitVec, false, {},
		1.0e-3f, 1.0e-3f, 5000, 5000);

	resultFit[0] = resultFitVec[0];
	resultFit[1] = resultFitVec[1];
	resultFit[2] = resultFitVec[2];
	resultFit[3] = resultFitVec[3];*/

	// Update LTC with best fitting values
	fitter.update(resultFit);
}
