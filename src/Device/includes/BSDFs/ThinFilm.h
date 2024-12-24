/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_THIN_FILM_H
#define DEVICE_THIN_FILM_H

#include "HostDeviceCommon/Material/Material.h"

// Evaluation XYZ sensitivity curves in Fourier space
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F eval_sensitivity(float opd, float shift)
{
    // Use Gaussian fits

    float phase = 2.0f * M_PI * opd * 1.0e-6f;

    float3 val = make_float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    float3 pos = make_float3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    float3 var = make_float3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);
    float3 xyz = val * hippt::sqrt(2.0f * M_PI * var) * hippt::cos(pos * phase + shift) * hippt::exp(-1.0f * var * phase * phase);

    xyz.x += 9.7470e-14f * sqrt(2.0f * M_PI * 4.5282e+09f) * cos(2.2399e+06f * phase + shift) * exp(-4.5282e+09f * phase * phase);

    return ColorRGB32F(xyz / 1.0685e-7f);
}

/**
 * Reference: * [1] [A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence, 
 *                   Belcour, Barla, 2017, Supplemental document] https://hal.science/hal-01518344v2/file/supp-mat-small%20(1).pdf
 */
HIPRT_HOST_DEVICE HIPRT_INLINE void fresnel_phase(float cos_theta_i,
    float eta1,
    float eta2, float kappa2,
    float& phi_par, float& phi_perp) 
{
    float sinThetaSqr = 1.0f - hippt::square(cos_theta_i);
    float A = hippt::square(eta2) * (1.0f - hippt::square(kappa2)) - hippt::square(eta1) * sinThetaSqr;
    float B = sqrt(hippt::square(A) + hippt::square(2 * hippt::square(eta2) * kappa2));
    float U = sqrt((A + B) / 2.0);
    float V = sqrt((B - A) / 2.0);

    float phi_perp_y = 2.0f * eta1 * V * cos_theta_i;
    float phi_perp_x = hippt::square(U) + hippt::square(V) - hippt::square(eta1 * cos_theta_i);
    phi_perp = atan2(phi_perp_y, phi_perp_x);

    float phi_par_y = 2.0f * eta1 * hippt::square(eta2) * cos_theta_i * (2 * kappa2 * U - (1.0f - hippt::square(kappa2)) * V);
    float phi_par_x = hippt::square(hippt::square(eta2) * (1.0f + hippt::square(kappa2)) * cos_theta_i) - hippt::square(eta1) * (hippt::square(U) + hippt::square(V));
    phi_par = atan2(phi_par_y, phi_par_x);
}

HIPRT_HOST_DEVICE HIPRT_INLINE void fresnel_conductor(float cos_theta_i,
    float eta, float k,
    float& Rp2, float& Rs2) 
{
    float cos_theta_i_2 = cos_theta_i * cos_theta_i;
    float sin_theta_i_2 = 1.0f - cos_theta_i_2;

    float temp1 = eta * eta - k * k - sin_theta_i_2;
    float a2pb2 = sqrt(temp1 * temp1 + 4.0f * k * k * eta * eta);
    float a = sqrt(0.5f * (a2pb2 + temp1));

    float term1 = a2pb2 + cos_theta_i_2;
    float term2 = 2.0f * a * cos_theta_i;

    Rs2 = (term1 - term2) / (term1 + term2);
    Rs2 = hippt::clamp(0.0f, 1.0f, Rs2);

    float term3 = a2pb2 * cos_theta_i_2 + sin_theta_i_2 * sin_theta_i_2;
    float term4 = term2 * sin_theta_i_2;

    Rp2 = Rs2 * (term3 - term4) / (term3 + term4);
    Rp2 = hippt::clamp(0.0f, 1.0f, Rp2);
}

/**
 * Reference: https://stackoverflow.com/questions/8507885/shift-hue-of-an-rgb-color
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F RGB_hue_shift(const ColorRGB32F& color, float hue_shift_degrees)
{
    if (hue_shift_degrees == 0.0f)
        return color;

    float cosA = cos(hue_shift_degrees / 180.0f * M_PI);
    float sinA = sin(hue_shift_degrees / 180.0f * M_PI);

    float3x3 matrix;
    matrix.m[0][0] = cosA + (1.0 - cosA) / 3.0;
    matrix.m[0][1] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA;
    matrix.m[0][2] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA;
    matrix.m[1][0] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA;
    matrix.m[1][1] = cosA + 1. / 3. * (1.0 - cosA);
    matrix.m[1][2] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA;
    matrix.m[2][0] = 1. / 3. * (1.0 - cosA) - sqrt(1. / 3.) * sinA;
    matrix.m[2][1] = 1. / 3. * (1.0 - cosA) + sqrt(1. / 3.) * sinA;
    matrix.m[2][2] = cosA + 1. / 3. * (1.0 - cosA);

    ColorRGB32F hue_shifted;
    hue_shifted.r = color.r * matrix.m[0][0] + color.g * matrix.m[0][1] + color.b * matrix.m[0][2];
    hue_shifted.g = color.r * matrix.m[1][0] + color.g * matrix.m[1][1] + color.b * matrix.m[1][2];
    hue_shifted.b = color.r * matrix.m[2][0] + color.g * matrix.m[2][1] + color.b * matrix.m[2][2];

    hue_shifted.clamp(0.0f, 1.0f);
    return hue_shifted;
}

/**
 * References:
 *
 * [1] [A Practical Extension to Microfacet Theory for the Modeling of Varying Iridescence, Belcour, Barla, 2017] https://belcour.github.io/blog/research/publication/2017/05/01/brdf-thin-film.html
 */
HIPRT_HOST_DEVICE HIPRT_INLINE ColorRGB32F thin_film_fresnel(const DeviceUnpackedEffectiveMaterial& material,
    float ambient_IOR, float HoL)
{
    float eta1 = ambient_IOR;
    float eta2 = material.thin_film_ior;
    float eta3 = material.thin_film_do_ior_override ? material.thin_film_base_ior_override : material.ior;
    // If override is not used, just default to 0.0f because the principled BSDF doesn't have 
    // complex IORs support anyways
    float kappa3 = material.thin_film_do_ior_override ? material.thin_film_kappa_3 : 0.0f;

    /* Compute the Spectral versions of the Fresnel reflectance and
     * transmitance for each interface. */
    float R12p = 0.0f;
    float R12s = 0.0f;
    float T121p = 0.0f;
    float T121s = 0.0f;
    float R23p = 0.0f;
    float R23s = 0.0f;
    float cos_theta_2 = 0.0f;

    float cos_theta_transmission_2 = 1.0f - (1.0f - hippt::square(HoL)) * hippt::square(eta1 / eta2);
    if (cos_theta_transmission_2 <= 0.0f)
    {
        // Total internal reflection
        R12s = 1.0f;
        R12p = 1.0f;

        // 0 transmission for total internal reflection
        T121p = 0.0f;
        T121s = 0.0f;
    }
    else
    {
        cos_theta_2 = sqrt(cos_theta_transmission_2);
        fresnel_conductor(HoL, eta2 / eta1, 0.0f, R12p, R12s);

        // Reflected part by the base
        fresnel_conductor(cos_theta_2, eta3 / eta2, kappa3, R23p, R23s);

        // Compute the transmission coefficients
        T121p = 1.0 - R12p;
        T121s = 1.0 - R12s;
    }

    /* Optical Path Difference */
    // float D = 2.0f * eta2 * film_thickness / 1000.0f * cos_theta_2;
    float D = material.thin_film_thickness / 1000.0f * cos_theta_2;

    /* Variables */
    float phi21p;
    float phi21s;
    float phi23p;
    float phi23s;

    /* Evaluate the phase shift */
    fresnel_phase(HoL, eta1, eta2, 0.0f, phi21p, phi21s);
    fresnel_phase(cos_theta_2, eta2, eta3, kappa3, phi23p, phi23s);
    phi21p = M_PI - phi21p;
    phi21s = M_PI - phi21s;

    float r123p = sqrt(R12p * R23p);
    float r123s = sqrt(R12s * R23s);

    /* Iridescence term using spectral antialiasing for Parallel polarization */
    // Reflectance term for m=0 (DC term amplitude)
    float Rs = (hippt::square(T121p) * R23p) / (1.0f - R12p * R23p);
    float C0 = R12p + Rs;
    
    ColorRGB32F I = ColorRGB32F(C0);
    ColorRGB32F Sm;

    // Reflectance term for m>0 (pairs of diracs)
    float Cm = Rs - T121p;
    for (int m = 1; m <= 2; ++m)
    {
        Cm *= r123p;
        Sm = 2.0f * eval_sensitivity(m * D, m * (phi23p + phi21p));
        I += Cm * Sm;
    }

    /* Iridescence term using spectral antialiasing for Perpendicular polarization */
    // Reflectance term for m=0 (DC term amplitude)
    Rs = (hippt::square(T121s) * R23s) / (1.0f - R12s * R23s);
    C0 = R12s + Rs;
    I += ColorRGB32F(C0);

    // Reflectance term for m>0 (pairs of diracs)
    Cm = Rs - T121s;
    for (int m = 1; m <= 2; ++m)
    {
        Cm *= r123s;
        Sm = 2.0f * eval_sensitivity(m * D, m * (phi23s + phi21s));
        I += Cm * Sm;
    }

    I *= 0.5f;

    // CIE RGB and CIE XYZ 1931 conversion:
    // source: https://en.wikipedia.org/wiki/CIE_1931_color_space
    float r = 2.3646381f * I[0] - 0.8965361f * I[1] - 0.4680737f * I[2];
    float g = -0.5151664f * I[0] + 1.4264000f * I[1] + 0.0887608f * I[2];
    float b = 0.0052037f * I[0] - 0.0144081f * I[1] + 1.0092106f * I[2];

    I = ColorRGB32F(r, g, b);
    I.clamp(0.0f, 1.0f);

    return RGB_hue_shift(I, material.thin_film_hue_shift_degrees * 360.0f);
}

#endif
