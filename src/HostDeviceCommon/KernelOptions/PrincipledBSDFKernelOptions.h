/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef HOST_DEVICE_COMMON_PRINCIPLED_BSDF_KERNEL_OPTIONS_H
#define HOST_DEVICE_COMMON_PRINCIPLED_BSDF_KERNEL_OPTIONS_H

#include "HostDeviceCommon/KernelOptions/Common.h"

/**
 * This file references the path tracer options that can be passed to HIPCC/NVCC using the -D <macro>=<value> option.
 * These path tracer options allow "compile-time" branching to enable/disable a variety
 * of functionalities in the path tracer.
 * 
 * For example, you can decide, at kernel compile-time, what envmap sampling strategy to use 
 *	- "CDF + Binary search"
 *	- "Alias table"
 * by passing the "-D EnvmapSamplingStrategy=1" or "-D EnvmapSamplingStrategy=2" option string during
 * the compilation of the kernel (for "CDF" and "alias table" respectively).
 * 
 * If you wish to change one of the option used by the path tracer at runtime (by interacting with
 * ImGui for example), you will have to recompile the kernel with the correct set of options
 * passed to the kernel compiler.
 * 
 * The advantage of recompiling the entire kernel over branching with a simple if() condition on
 * a variable (that would be passed in RenderData for example) is that the recompiling approach
 * does not incur an additional register cost that would harm the occupancy potential of the kernel
 * (whereas registers may be allocated for the block {} of the if() conditions since the compiler
 * has no way to know which branch of the if is going to be taken at runtime).
 */

/**
 * Those are simple defines to give names to the option values.
 * This allows the use of LSS_ONE_RANDOM_LIGHT_MIS (for example) instead of a hardcoded '2'
 */
#define PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN 0
#define PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR 1

#define GGX_VNDF_SAMPLING 0
#define GGX_VNDF_SPHERICAL_CAPS 1
#define GGX_VNDF_BOUNDED 2

/**
 * Options are defined in a #ifndef __KERNELCC__ block because:
 *	- If they were not, the would be defined on the GPU side. However, the -D <macro>=<value> compiler option
 *		cannot override a #define statement. This means that if the #define statement are encountered by the compiler,
 *		we cannot modify the value of the macros anymore with the -D option which means no run-time switching / experimenting :(
 * - The CPU still needs the options to be able to compile the code so here they are, in a CPU-only block
 */
#ifndef __KERNELCC__

/**
 * What diffuse lobe to use in the principled BSDF.
 * 
 *	- PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN
 *		Use a lambertian BRDF for the diffuse lobe
 * 
 *	- PRINCIPLED_DIFFUSE_LOBE_OREN_NAYAR
 *		Use an Oren-Nayar BRDF for the diffuse lobe
 */
#define PrincipledBSDFDiffuseLobe PRINCIPLED_DIFFUSE_LOBE_LAMBERTIAN

/**
 * What sampling strategy to use for the GGX NDF
 *
 *  - GGX_NO_VNDF [Not Yet Implemented]
 *		Not sampling the visible distribution of normals.
 *		Just classic GGX sampling
 *
 *  - GGX_VNDF_SAMPLING
 *		Sample the distribution of visible normals as proposed
 *		in [Sampling the GGX Distribution of Visible Normals, Heitz, 2018]
 *
 *  - GGX_VNDF_SPHERICAL_CAPS
 *		Sample the distribution of visible normals using spherical
 *		caps as proposed in [Sampling Visible GGX Normals with Spherical Caps, Dupuy & Benyoub, 2023]
 *
 *  - GGX_VNDF_BOUNDED [Not Yet Implemented]
 *		Sample the distribution of visible normals with a bounded VNDF
 *		sampling range as proposed in [Bounded VNDF Sampling for Smith-GGX Reflections, Eto & Tokuyoshi, 2023]
 *
 */
#define PrincipledBSDFAnisotropicGGXSampleFunction GGX_VNDF_SAMPLING

/**
 * Whether or not to use multiple scattering to conserve energy when evaluating
 * GGX BRDF lobes in the Principled BSDF
 * 
 * This is implemented by following 
 * [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
 * 
 * Possible options are KERNEL_OPTION_TRUE and KERNEL_OPTION_FALSE. Self explanatory.
 */
#define PrincipledBSDFDoEnergyCompensation KERNEL_OPTION_TRUE

/**
 * If KERNEL_OPTION_TRUE, on-the-fly monte carlo integration of the clearcoat lobe
 * directional albedo will be performed to ensure energy conservation & preservation
 * of a material with a clearcoat lobe
 */
#define PrincipledBSDFEnforceStrongEnergyConservation KERNEL_OPTION_FALSE

/**
 * Whether or not to perform energy compensation for the glass layer of the Principled BSDF
 */
#define PrincipledBSDFDoGlassEnergyCompensation KERNEL_OPTION_TRUE

/**
 * Whether or not to perform energy compensation (it's an approximation for the clearcoat
 * layer, it's not perfect but very good in most cases) for the clearcoat layer of the Principled BSDF
 */
#define PrincipledBSDFDoClearcoatEnergyCompensation KERNEL_OPTION_TRUE

/**
 * Whether or not to perform energy compensation for the metallic layer of the Principled BSDF
 */
#define PrincipledBSDFDoMetallicEnergyCompensation KERNEL_OPTION_TRUE

/**
 * Whether or not to use multiple scattering to conserve energy and use a
 * Fresnel compensation term i.e. account for Fresnel when light scatters multiple
 * times on the microsurface. This increases saturation and has a noticeable impact.
 * Only applies to conductors. This term always is implicitely used for dielectrics
 *
 * This is implemented by following
 * [Practical multiple scattering compensation for microfacet models, Turquin, 2019]
 *
 * Possible options are KERNEL_OPTION_TRUE and KERNEL_OPTION_FALSE. Self explanatory.
 */
#define PrincipledBSDFDoMetallicFresnelEnergyCompensation KERNEL_OPTION_TRUE

/**
 * Whether or not to perform energy compensation for the specular/diffuse layer of the Principled BSDF
 */
#define PrincipledBSDFDoSpecularEnergyCompensation KERNEL_OPTION_TRUE

#endif // #ifndef __KERNELCC__

#endif