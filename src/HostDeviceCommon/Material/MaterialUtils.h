/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_MATERIAL_UTILS_H
#define HOST_DEVICE_COMMON_MATERIAL_UTILS_H

struct MaterialUtils
{
    static constexpr int NO_TEXTURE = 65535;
    // When an emissive texture is read and is determine to be
    // constant, no emissive texture will be used. Instead,
    // we'll just set the emission of the material to that constant emission value
    // and the emissive texture index of the material will be replaced by
    // CONSTANT_EMISSIVE_TEXTURE
    static constexpr int CONSTANT_EMISSIVE_TEXTURE = 65534;
    // Maximum number of different textures per scene
    static constexpr int MAX_TEXTURE_COUNT = 65533;

    static constexpr float ROUGHNESS_CLAMP = 1.0e-4f;

    HIPRT_HOST_DEVICE static void get_oren_nayar_AB(float sigma, float& out_oren_A, float& out_oren_B)
    {
        float sigma2 = sigma * sigma;
        out_oren_A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
        out_oren_B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    HIPRT_HOST_DEVICE static void get_alphas(float roughness, float anisotropy, float& out_alpha_x, float& out_alpha_y)
    {
        float aspect = sqrtf(1.0f - 0.9f * anisotropy);
        out_alpha_x = hippt::max(ROUGHNESS_CLAMP, roughness * roughness / aspect);
        out_alpha_y = hippt::max(ROUGHNESS_CLAMP, roughness * roughness * aspect);
    }

    HIPRT_HOST_DEVICE static float get_thin_walled_roughness(bool thin_walled, float base_roughness, float relative_eta)
    {
        if (!thin_walled)
            return base_roughness;

        /*
         * Roughness remapping so that a thin walled interface matches better a
         * properly modeled double interface model. Said otherwise: roughness remapping
         * so that the thin walled approximation matches the non thin walled physically correct equivalent
         *
         * Reference:
         * [Revisiting Physically Based Shading at Imageworks, Christopher Kulla & Alejandro Conty, 2017]
         *
         * https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
         */
        float remapped = base_roughness * sqrt(3.7f * (relative_eta - 1.0f) * hippt::square(relative_eta - 0.5f) / hippt::pow_3(relative_eta));

        // Remapped roughness starts going above 1.0f starting at relative eta around 1.9f
        // and ends up at 1.39f at relative eta 3.5f
        //
        // Because we don't expect the user to input higher IOR values than that,
        // we remap that remapped roughness from [0.0f, 1.39f] to [0.0f, 1.0f]
        // and if the user inputs higher IOR values than 3.5f, we clamp to 1.0f roughness
        // anyways
        return hippt::clamp(0.0f, 1.0f, remapped / 1.39f);
    }
};

#endif
