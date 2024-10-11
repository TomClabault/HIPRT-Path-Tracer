/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef DEVICE_INCLUDES_BSDFS_SHEEN_LTC_PARAMETERS
#define DEVICE_INCLUDES_BSDFS_SHEEN_LTC_PARAMETERS

#ifndef __KERNELCC__
// This file should not be included on the GPU

 /**
  * Reference:
  *
  * [1]: [Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines - Github] [https://github.com/tizian/ltc-sheen/blob/master/fitting/python/data/ltc_table_sheen_approx.cpp]
  */

#include "HostDeviceCommon/Math.h"

#include <array>

  /**
   * Precomputed parameters for the fitted LTC (Linearly Transformed Cosine)
   * distribution of an analytic approximation of the reference volumetric SGGX
   * sheen layer.
   *
   * Sampled as [y][x] = float3(Ai, Bi, Ri) with:
   *  y = cos(theta)
   *  x = alpha
   */

static std::array<float3, 32*32> ltc_parameters_table_approximation = {

        make_float3(0.10027f, -0.00000f, 0.33971f), make_float3(0.10760f, -0.00000f, 0.35542f), make_float3(0.11991f, 0.00001f, 0.30888f),
        make_float3(0.13148f, 0.00001f, 0.23195f), make_float3(0.14227f, 0.00001f, 0.15949f), make_float3(0.15231f, -0.00000f, 0.10356f),
        make_float3(0.16168f, -0.00000f, 0.06466f), make_float3(0.17044f, 0.00000f, 0.03925f), make_float3(0.17867f, 0.00001f, 0.02334f),
        make_float3(0.18645f, 0.00000f, 0.01366f), make_float3(0.19382f, -0.00000f, 0.00790f), make_float3(0.20084f, -0.00001f, 0.00452f),
        make_float3(0.20754f, 0.00001f, 0.00257f), make_float3(0.21395f, 0.00000f, 0.00145f), make_float3(0.22011f, 0.00000f, 0.00081f),
        make_float3(0.22603f, -0.00000f, 0.00045f), make_float3(0.23174f, -0.00001f, 0.00025f), make_float3(0.23726f, 0.00000f, 0.00014f),
        make_float3(0.24259f, -0.00001f, 0.00008f), make_float3(0.24777f, -0.00001f, 0.00004f), make_float3(0.25279f, -0.00001f, 0.00002f),
        make_float3(0.25768f, 0.00001f, 0.00001f), make_float3(0.26243f, 0.00001f, 0.00001f), make_float3(0.26707f, -0.00000f, 0.00000f),
        make_float3(0.27159f, -0.00000f, 0.00000f), make_float3(0.27601f, -0.00000f, 0.00000f), make_float3(0.28033f, 0.00000f, 0.00000f),
        make_float3(0.28456f, -0.00000f, 0.00000f), make_float3(0.28870f, -0.00001f, 0.00000f), make_float3(0.29276f, -0.00000f, 0.00000f),
        make_float3(0.29676f, 0.00000f, 0.00000f), make_float3(0.30067f, -0.00001f, 0.00000f),


        make_float3(0.10068f, -0.00013f, 0.33844f), make_float3(0.10802f, -0.00001f, 0.35438f), make_float3(0.12031f, -0.00000f, 0.30859f),
        make_float3(0.13190f, 0.00001f, 0.23230f), make_float3(0.14269f, 0.00000f, 0.16016f), make_float3(0.15274f, 0.00001f, 0.10429f),
        make_float3(0.16211f, -0.00000f, 0.06529f), make_float3(0.17088f, -0.00000f, 0.03975f), make_float3(0.17912f, -0.00000f, 0.02370f),
        make_float3(0.18691f, -0.00001f, 0.01391f), make_float3(0.19429f, -0.00000f, 0.00807f), make_float3(0.20132f, -0.00000f, 0.00463f),
        make_float3(0.20803f, 0.00001f, 0.00264f), make_float3(0.21445f, 0.00000f, 0.00149f), make_float3(0.22061f, 0.00000f, 0.00084f),
        make_float3(0.22655f, -0.00000f, 0.00047f), make_float3(0.23226f, -0.00000f, 0.00026f), make_float3(0.23779f, -0.00001f, 0.00015f),
        make_float3(0.24314f, -0.00002f, 0.00008f), make_float3(0.24832f, -0.00001f, 0.00004f), make_float3(0.25336f, -0.00002f, 0.00002f),
        make_float3(0.25824f, -0.00003f, 0.00001f), make_float3(0.26301f, -0.00005f, 0.00001f), make_float3(0.26766f, -0.00008f, 0.00000f),
        make_float3(0.27220f, -0.00012f, 0.00000f), make_float3(0.27665f, -0.00019f, 0.00000f), make_float3(0.28101f, -0.00026f, 0.00000f),
        make_float3(0.28532f, -0.00041f, 0.00000f), make_float3(0.28960f, -0.00058f, 0.00000f), make_float3(0.29389f, -0.00080f, 0.00000f),
        make_float3(0.29830f, -0.00096f, 0.00000f), make_float3(0.30309f, 0.00000f, 0.00000f),


        make_float3(0.09988f, -0.00743f, 0.33595f), make_float3(0.10928f, -0.00078f, 0.35135f), make_float3(0.12169f, -0.00015f, 0.30790f),
        make_float3(0.13330f, -0.00006f, 0.23367f), make_float3(0.14412f, -0.00003f, 0.16253f), make_float3(0.15420f, -0.00004f, 0.10680f),
        make_float3(0.16360f, -0.00002f, 0.06750f), make_float3(0.17240f, -0.00004f, 0.04148f), make_float3(0.18068f, -0.00003f, 0.02497f),
        make_float3(0.18850f, -0.00004f, 0.01480f), make_float3(0.19591f, -0.00005f, 0.00867f), make_float3(0.20297f, -0.00005f, 0.00503f),
        make_float3(0.20970f, -0.00008f, 0.00289f), make_float3(0.21616f, -0.00012f, 0.00165f), make_float3(0.22235f, -0.00015f, 0.00094f),
        make_float3(0.22831f, -0.00020f, 0.00053f), make_float3(0.23407f, -0.00028f, 0.00030f), make_float3(0.23963f, -0.00039f, 0.00017f),
        make_float3(0.24503f, -0.00056f, 0.00009f), make_float3(0.25028f, -0.00084f, 0.00005f), make_float3(0.25541f, -0.00124f, 0.00003f),
        make_float3(0.26045f, -0.00184f, 0.00002f), make_float3(0.26545f, -0.00274f, 0.00001f), make_float3(0.27049f, -0.00410f, 0.00001f),
        make_float3(0.27569f, -0.00612f, 0.00000f), make_float3(0.28127f, -0.00908f, 0.00000f), make_float3(0.28762f, -0.01335f, 0.00000f),
        make_float3(0.29542f, -0.01917f, 0.00000f), make_float3(0.30593f, -0.02649f, 0.00000f), make_float3(0.32153f, -0.03397f, 0.00000f),
        make_float3(0.34697f, -0.03669f, 0.00000f), make_float3(0.39704f, -0.00000f, 0.00000f),


        make_float3(0.08375f, -0.07516f, 0.33643f), make_float3(0.10999f, -0.00776f, 0.34873f), make_float3(0.12392f, -0.00160f, 0.30771f),
        make_float3(0.13571f, -0.00070f, 0.23660f), make_float3(0.14661f, -0.00044f, 0.16701f), make_float3(0.15676f, -0.00034f, 0.11147f),
        make_float3(0.16621f, -0.00032f, 0.07157f), make_float3(0.17508f, -0.00029f, 0.04470f), make_float3(0.18341f, -0.00033f, 0.02736f),
        make_float3(0.19128f, -0.00038f, 0.01648f), make_float3(0.19876f, -0.00045f, 0.00981f), make_float3(0.20587f, -0.00057f, 0.00579f),
        make_float3(0.21267f, -0.00074f, 0.00339f), make_float3(0.21918f, -0.00096f, 0.00197f), make_float3(0.22544f, -0.00129f, 0.00114f),
        make_float3(0.23150f, -0.00179f, 0.00066f), make_float3(0.23736f, -0.00249f, 0.00038f), make_float3(0.24308f, -0.00352f, 0.00022f),
        make_float3(0.24871f, -0.00501f, 0.00012f), make_float3(0.25433f, -0.00721f, 0.00007f), make_float3(0.26003f, -0.01042f, 0.00004f),
        make_float3(0.26604f, -0.01512f, 0.00002f), make_float3(0.27264f, -0.02192f, 0.00001f), make_float3(0.28039f, -0.03158f, 0.00001f),
        make_float3(0.29024f, -0.04491f, 0.00000f), make_float3(0.30382f, -0.06232f, 0.00000f), make_float3(0.32395f, -0.08307f, 0.00000f),
        make_float3(0.35533f, -0.10404f, 0.00000f), make_float3(0.40491f, -0.11883f, 0.00000f), make_float3(0.47816f, -0.11902f, 0.00000f),
        make_float3(0.56774f, -0.09644f, 0.00000f), make_float3(0.66332f, -0.00000f, 0.00000f),


        make_float3(0.05655f, -0.31167f, 0.32420f), make_float3(0.10687f, -0.03376f, 0.35090f), make_float3(0.12655f, -0.00804f, 0.31009f),
        make_float3(0.13914f, -0.00363f, 0.24233f), make_float3(0.15029f, -0.00227f, 0.17455f), make_float3(0.16057f, -0.00177f, 0.11907f),
        make_float3(0.17014f, -0.00156f, 0.07820f), make_float3(0.17910f, -0.00153f, 0.04999f), make_float3(0.18752f, -0.00162f, 0.03132f),
        make_float3(0.19549f, -0.00182f, 0.01932f), make_float3(0.20304f, -0.00213f, 0.01178f), make_float3(0.21025f, -0.00261f, 0.00712f),
        make_float3(0.21716f, -0.00326f, 0.00427f), make_float3(0.22380f, -0.00420f, 0.00255f), make_float3(0.23022f, -0.00554f, 0.00151f),
        make_float3(0.23648f, -0.00740f, 0.00089f), make_float3(0.24264f, -0.01005f, 0.00053f), make_float3(0.24879f, -0.01381f, 0.00031f),
        make_float3(0.25510f, -0.01911f, 0.00018f), make_float3(0.26177f, -0.02658f, 0.00011f), make_float3(0.26918f, -0.03698f, 0.00006f),
        make_float3(0.27795f, -0.05122f, 0.00004f), make_float3(0.28910f, -0.07004f, 0.00002f), make_float3(0.30427f, -0.09359f, 0.00001f),
        make_float3(0.32606f, -0.12066f, 0.00001f), make_float3(0.35822f, -0.14764f, 0.00001f), make_float3(0.40512f, -0.16863f, 0.00000f),
        make_float3(0.46849f, -0.17766f, 0.00000f), make_float3(0.54169f, -0.17178f, 0.00000f), make_float3(0.61239f, -0.15052f, 0.00000f),
        make_float3(0.67350f, -0.11117f, 0.00000f), make_float3(0.73152f, 0.00000f, 0.00000f),


        make_float3(0.05336f, -0.34864f, 0.38172f), make_float3(0.09920f, -0.08509f, 0.36009f), make_float3(0.12900f, -0.02477f, 0.31816f),
        make_float3(0.14348f, -0.01195f, 0.25287f), make_float3(0.15525f, -0.00763f, 0.18668f), make_float3(0.16584f, -0.00587f, 0.13092f),
        make_float3(0.17560f, -0.00512f, 0.08853f), make_float3(0.18472f, -0.00492f, 0.05832f), make_float3(0.19329f, -0.00508f, 0.03768f),
        make_float3(0.20140f, -0.00555f, 0.02399f), make_float3(0.20910f, -0.00634f, 0.01510f), make_float3(0.21647f, -0.00749f, 0.00942f),
        make_float3(0.22355f, -0.00914f, 0.00584f), make_float3(0.23039f, -0.01140f, 0.00360f), make_float3(0.23709f, -0.01450f, 0.00221f),
        make_float3(0.24371f, -0.01877f, 0.00135f), make_float3(0.25039f, -0.02458f, 0.00083f), make_float3(0.25730f, -0.03248f, 0.00051f),
        make_float3(0.26474f, -0.04313f, 0.00031f), make_float3(0.27313f, -0.05721f, 0.00019f), make_float3(0.28319f, -0.07550f, 0.00012f),
        make_float3(0.29598f, -0.09825f, 0.00008f), make_float3(0.31310f, -0.12490f, 0.00005f), make_float3(0.33683f, -0.15332f, 0.00003f),
        make_float3(0.36995f, -0.17964f, 0.00002f), make_float3(0.41497f, -0.19882f, 0.00002f), make_float3(0.47174f, -0.20686f, 0.00001f),
        make_float3(0.53490f, -0.20242f, 0.00001f), make_float3(0.59635f, -0.18603f, 0.00001f), make_float3(0.65092f, -0.15783f, 0.00001f),
        make_float3(0.69798f, -0.11426f, 0.00001f), make_float3(0.74494f, 0.00000f, 0.00001f),


        make_float3(0.05749f, -0.31793f, 0.44455f), make_float3(0.09398f, -0.14133f, 0.37908f), make_float3(0.13152f, -0.05344f, 0.33487f),
        make_float3(0.14884f, -0.02831f, 0.27078f), make_float3(0.16170f, -0.01861f, 0.20554f), make_float3(0.17282f, -0.01431f, 0.14892f),
        make_float3(0.18293f, -0.01233f, 0.10431f), make_float3(0.19231f, -0.01159f, 0.07128f), make_float3(0.20110f, -0.01165f, 0.04782f),
        make_float3(0.20942f, -0.01233f, 0.03163f), make_float3(0.21733f, -0.01364f, 0.02070f), make_float3(0.22491f, -0.01559f, 0.01344f),
        make_float3(0.23224f, -0.01830f, 0.00867f), make_float3(0.23938f, -0.02200f, 0.00557f), make_float3(0.24643f, -0.02692f, 0.00357f),
        make_float3(0.25351f, -0.03342f, 0.00228f), make_float3(0.26079f, -0.04192f, 0.00146f), make_float3(0.26852f, -0.05298f, 0.00094f),
        make_float3(0.27707f, -0.06708f, 0.00060f), make_float3(0.28698f, -0.08468f, 0.00039f), make_float3(0.29907f, -0.10591f, 0.00026f),
        make_float3(0.31446f, -0.13028f, 0.00017f), make_float3(0.33466f, -0.15629f, 0.00012f), make_float3(0.36155f, -0.18122f, 0.00008f),
        make_float3(0.39700f, -0.20145f, 0.00006f), make_float3(0.44194f, -0.21352f, 0.00005f), make_float3(0.49484f, -0.21535f, 0.00004f),
        make_float3(0.55114f, -0.20661f, 0.00003f), make_float3(0.60550f, -0.18774f, 0.00003f), make_float3(0.65477f, -0.15830f, 0.00002f),
        make_float3(0.69863f, -0.11427f, 0.00002f), make_float3(0.74332f, 0.00000f, 0.00002f),


        make_float3(0.06502f, -0.28106f, 0.49493f), make_float3(0.09745f, -0.17506f, 0.41308f), make_float3(0.13592f, -0.08778f, 0.36191f),
        make_float3(0.15585f, -0.05167f, 0.29839f), make_float3(0.17008f, -0.03538f, 0.23353f), make_float3(0.18198f, -0.02741f, 0.17549f),
        make_float3(0.19260f, -0.02338f, 0.12792f), make_float3(0.20235f, -0.02153f, 0.09115f), make_float3(0.21146f, -0.02104f, 0.06384f),
        make_float3(0.22006f, -0.02158f, 0.04414f), make_float3(0.22825f, -0.02299f, 0.03022f), make_float3(0.23612f, -0.02528f, 0.02053f),
        make_float3(0.24374f, -0.02852f, 0.01387f), make_float3(0.25120f, -0.03287f, 0.00933f), make_float3(0.25860f, -0.03853f, 0.00626f),
        make_float3(0.26608f, -0.04579f, 0.00420f), make_float3(0.27381f, -0.05493f, 0.00282f), make_float3(0.28203f, -0.06627f, 0.00189f),
        make_float3(0.29109f, -0.08016f, 0.00128f), make_float3(0.30146f, -0.09668f, 0.00087f), make_float3(0.31381f, -0.11579f, 0.00060f),
        make_float3(0.32900f, -0.13678f, 0.00042f), make_float3(0.34816f, -0.15845f, 0.00030f), make_float3(0.37257f, -0.17874f, 0.00022f),
        make_float3(0.40358f, -0.19512f, 0.00016f), make_float3(0.44202f, -0.20502f, 0.00013f), make_float3(0.48743f, -0.20660f, 0.00010f),
        make_float3(0.53748f, -0.19897f, 0.00008f), make_float3(0.58855f, -0.18183f, 0.00007f), make_float3(0.63751f, -0.15424f, 0.00006f),
        make_float3(0.68300f, -0.11197f, 0.00005f), make_float3(0.72978f, 0.00001f, 0.00005f),


        make_float3(0.07528f, -0.24711f, 0.53455f), make_float3(0.10863f, -0.18975f, 0.45764f), make_float3(0.14433f, -0.11815f, 0.39949f),
        make_float3(0.16570f, -0.07688f, 0.33687f), make_float3(0.18121f, -0.05511f, 0.27245f), make_float3(0.19398f, -0.04327f, 0.21288f),
        make_float3(0.20521f, -0.03673f, 0.16193f), make_float3(0.21545f, -0.03319f, 0.12068f), make_float3(0.22496f, -0.03159f, 0.08855f),
        make_float3(0.23392f, -0.03136f, 0.06420f), make_float3(0.24244f, -0.03223f, 0.04612f), make_float3(0.25063f, -0.03408f, 0.03290f),
        make_float3(0.25857f, -0.03690f, 0.02335f), make_float3(0.26634f, -0.04074f, 0.01650f), make_float3(0.27403f, -0.04570f, 0.01163f),
        make_float3(0.28178f, -0.05195f, 0.00819f), make_float3(0.28972f, -0.05961f, 0.00577f), make_float3(0.29802f, -0.06886f, 0.00406f),
        make_float3(0.30695f, -0.07982f, 0.00287f), make_float3(0.31682f, -0.09255f, 0.00204f), make_float3(0.32809f, -0.10692f, 0.00146f),
        make_float3(0.34129f, -0.12259f, 0.00105f), make_float3(0.35714f, -0.13882f, 0.00077f), make_float3(0.37647f, -0.15448f, 0.00057f),
        make_float3(0.40023f, -0.16805f, 0.00043f), make_float3(0.42931f, -0.17770f, 0.00033f), make_float3(0.46430f, -0.18158f, 0.00026f),
        make_float3(0.50501f, -0.17811f, 0.00021f), make_float3(0.55015f, -0.16592f, 0.00018f), make_float3(0.59760f, -0.14331f, 0.00015f),
        make_float3(0.64549f, -0.10572f, 0.00013f), make_float3(0.69674f, 0.00000f, 0.00012f),


        make_float3(0.08968f, -0.22166f, 0.57013f), make_float3(0.12485f, -0.19606f, 0.50498f), make_float3(0.15755f, -0.13952f, 0.44492f),
        make_float3(0.17936f, -0.09867f, 0.38452f), make_float3(0.19578f, -0.07393f, 0.32154f), make_float3(0.20935f, -0.05908f, 0.26116f),
        make_float3(0.22123f, -0.05009f, 0.20725f), make_float3(0.23199f, -0.04467f, 0.16152f), make_float3(0.24197f, -0.04157f, 0.12414f),
        make_float3(0.25134f, -0.04011f, 0.09438f), make_float3(0.26024f, -0.03986f, 0.07115f), make_float3(0.26879f, -0.04064f, 0.05329f),
        make_float3(0.27706f, -0.04233f, 0.03970f), make_float3(0.28514f, -0.04488f, 0.02947f), make_float3(0.29311f, -0.04830f, 0.02181f),
        make_float3(0.30105f, -0.05263f, 0.01611f), make_float3(0.30908f, -0.05790f, 0.01189f), make_float3(0.31731f, -0.06417f, 0.00877f),
        make_float3(0.32591f, -0.07150f, 0.00648f), make_float3(0.33507f, -0.07986f, 0.00480f), make_float3(0.34502f, -0.08921f, 0.00356f),
        make_float3(0.35610f, -0.09937f, 0.00266f), make_float3(0.36867f, -0.11001f, 0.00200f), make_float3(0.38323f, -0.12062f, 0.00151f),
        make_float3(0.40031f, -0.13041f, 0.00116f), make_float3(0.42056f, -0.13839f, 0.00090f), make_float3(0.44465f, -0.14322f, 0.00070f),
        make_float3(0.47318f, -0.14336f, 0.00056f), make_float3(0.50652f, -0.13700f, 0.00046f), make_float3(0.54470f, -0.12166f, 0.00038f),
        make_float3(0.58750f, -0.09227f, 0.00032f), make_float3(0.63790f, 0.00000f, 0.00028f),


        make_float3(0.10928f, -0.20982f, 0.60377f), make_float3(0.14371f, -0.19861f, 0.54677f), make_float3(0.17437f, -0.15359f, 0.48934f),
        make_float3(0.19613f, -0.11610f, 0.43261f), make_float3(0.21313f, -0.09068f, 0.37256f), make_float3(0.22735f, -0.07402f, 0.31310f),
        make_float3(0.23983f, -0.06310f, 0.25792f), make_float3(0.25113f, -0.05592f, 0.20914f), make_float3(0.26159f, -0.05128f, 0.16750f),
        make_float3(0.27141f, -0.04843f, 0.13286f), make_float3(0.28075f, -0.04692f, 0.10458f), make_float3(0.28970f, -0.04644f, 0.08182f),
        make_float3(0.29835f, -0.04682f, 0.06371f), make_float3(0.30679f, -0.04799f, 0.04942f), make_float3(0.31508f, -0.04984f, 0.03822f),
        make_float3(0.32329f, -0.05234f, 0.02949f), make_float3(0.33149f, -0.05550f, 0.02272f), make_float3(0.33979f, -0.05931f, 0.01749f),
        make_float3(0.34827f, -0.06373f, 0.01346f), make_float3(0.35704f, -0.06873f, 0.01036f), make_float3(0.36626f, -0.07428f, 0.00799f),
        make_float3(0.37608f, -0.08028f, 0.00617f), make_float3(0.38672f, -0.08655f, 0.00478f), make_float3(0.39843f, -0.09283f, 0.00371f),
        make_float3(0.41148f, -0.09873f, 0.00290f), make_float3(0.42626f, -0.10371f, 0.00228f), make_float3(0.44316f, -0.10703f, 0.00181f),
        make_float3(0.46265f, -0.10766f, 0.00145f), make_float3(0.48525f, -0.10415f, 0.00117f), make_float3(0.51155f, -0.09426f, 0.00096f),
        make_float3(0.54238f, -0.07326f, 0.00080f), make_float3(0.58108f, 0.00000f, 0.00068f),


        make_float3(0.13051f, -0.20682f, 0.62432f), make_float3(0.16330f, -0.20073f, 0.57323f), make_float3(0.19267f, -0.16509f, 0.51965f),
        make_float3(0.21421f, -0.13200f, 0.46714f), make_float3(0.23154f, -0.10730f, 0.41130f), make_float3(0.24629f, -0.08978f, 0.35487f),
        make_float3(0.25931f, -0.07747f, 0.30102f), make_float3(0.27115f, -0.06883f, 0.25188f), make_float3(0.28213f, -0.06278f, 0.20849f),
        make_float3(0.29244f, -0.05862f, 0.17111f), make_float3(0.30225f, -0.05585f, 0.13949f), make_float3(0.31166f, -0.05417f, 0.11309f),
        make_float3(0.32077f, -0.05341f, 0.09129f), make_float3(0.32964f, -0.05335f, 0.07343f), make_float3(0.33834f, -0.05393f, 0.05890f),
        make_float3(0.34693f, -0.05508f, 0.04713f), make_float3(0.35546f, -0.05671f, 0.03765f), make_float3(0.36403f, -0.05880f, 0.03004f),
        make_float3(0.37268f, -0.06133f, 0.02395f), make_float3(0.38149f, -0.06421f, 0.01909f), make_float3(0.39055f, -0.06741f, 0.01521f),
        make_float3(0.39997f, -0.07082f, 0.01213f), make_float3(0.40987f, -0.07434f, 0.00969f), make_float3(0.42039f, -0.07779f, 0.00775f),
        make_float3(0.43169f, -0.08092f, 0.00622f), make_float3(0.44398f, -0.08341f, 0.00500f), make_float3(0.45749f, -0.08480f, 0.00404f),
        make_float3(0.47249f, -0.08439f, 0.00328f), make_float3(0.48934f, -0.08117f, 0.00268f), make_float3(0.50847f, -0.07344f, 0.00220f),
        make_float3(0.53061f, -0.05739f, 0.00183f), make_float3(0.55814f, -0.00000f, 0.00155f),


        make_float3(0.15121f, -0.20607f, 0.62947f), make_float3(0.18300f, -0.20354f, 0.58335f), make_float3(0.21155f, -0.17590f, 0.53312f),
        make_float3(0.23282f, -0.14755f, 0.48447f), make_float3(0.25031f, -0.12455f, 0.43318f), make_float3(0.26543f, -0.10703f, 0.38096f),
        make_float3(0.27895f, -0.09394f, 0.33029f), make_float3(0.29131f, -0.08418f, 0.28305f), make_float3(0.30281f, -0.07693f, 0.24033f),
        make_float3(0.31365f, -0.07157f, 0.20255f), make_float3(0.32398f, -0.06766f, 0.16971f), make_float3(0.33391f, -0.06487f, 0.14151f),
        make_float3(0.34352f, -0.06300f, 0.11754f), make_float3(0.35289f, -0.06189f, 0.09733f), make_float3(0.36206f, -0.06138f, 0.08038f),
        make_float3(0.37112f, -0.06139f, 0.06624f), make_float3(0.38010f, -0.06186f, 0.05449f), make_float3(0.38905f, -0.06272f, 0.04477f),
        make_float3(0.39804f, -0.06390f, 0.03675f), make_float3(0.40711f, -0.06534f, 0.03014f), make_float3(0.41635f, -0.06699f, 0.02471f),
        make_float3(0.42580f, -0.06874f, 0.02026f), make_float3(0.43556f, -0.07051f, 0.01662f), make_float3(0.44571f, -0.07214f, 0.01365f),
        make_float3(0.45636f, -0.07346f, 0.01122f), make_float3(0.46763f, -0.07423f, 0.00924f), make_float3(0.47968f, -0.07408f, 0.00762f),
        make_float3(0.49266f, -0.07254f, 0.00631f), make_float3(0.50681f, -0.06883f, 0.00524f), make_float3(0.52242f, -0.06160f, 0.00437f),
        make_float3(0.54001f, -0.04778f, 0.00367f), make_float3(0.56112f, 0.00000f, 0.00312f),


        make_float3(0.17197f, -0.20561f, 0.62587f), make_float3(0.20338f, -0.20599f, 0.58408f), make_float3(0.23149f, -0.18515f, 0.53684f),
        make_float3(0.25251f, -0.16154f, 0.49150f), make_float3(0.27004f, -0.14094f, 0.44437f), make_float3(0.28542f, -0.12426f, 0.39642f),
        make_float3(0.29931f, -0.11107f, 0.34949f), make_float3(0.31213f, -0.10073f, 0.30512f), make_float3(0.32413f, -0.09263f, 0.26430f),
        make_float3(0.33550f, -0.08631f, 0.22751f), make_float3(0.34636f, -0.08139f, 0.19485f), make_float3(0.35682f, -0.07760f, 0.16619f),
        make_float3(0.36695f, -0.07471f, 0.14127f), make_float3(0.37684f, -0.07258f, 0.11976f), make_float3(0.38652f, -0.07106f, 0.10128f),
        make_float3(0.39607f, -0.07006f, 0.08549f), make_float3(0.40551f, -0.06949f, 0.07205f), make_float3(0.41490f, -0.06928f, 0.06064f),
        make_float3(0.42427f, -0.06933f, 0.05099f), make_float3(0.43369f, -0.06961f, 0.04283f), make_float3(0.44319f, -0.07004f, 0.03597f),
        make_float3(0.45283f, -0.07053f, 0.03019f), make_float3(0.46265f, -0.07097f, 0.02535f), make_float3(0.47273f, -0.07127f, 0.02128f),
        make_float3(0.48314f, -0.07125f, 0.01788f), make_float3(0.49395f, -0.07070f, 0.01504f), make_float3(0.50527f, -0.06936f, 0.01267f),
        make_float3(0.51721f, -0.06683f, 0.01069f), make_float3(0.52993f, -0.06247f, 0.00904f), make_float3(0.54362f, -0.05516f, 0.00766f),
        make_float3(0.55866f, -0.04229f, 0.00653f), make_float3(0.57607f, -0.00000f, 0.00561f),


        make_float3(0.19561f, -0.20369f, 0.62676f), make_float3(0.22742f, -0.20521f, 0.59059f), make_float3(0.25590f, -0.18887f, 0.54809f),
        make_float3(0.27707f, -0.16911f, 0.50677f), make_float3(0.29477f, -0.15102f, 0.46369f), make_float3(0.31041f, -0.13567f, 0.41959f),
        make_float3(0.32465f, -0.12298f, 0.37594f), make_float3(0.33788f, -0.11258f, 0.33408f), make_float3(0.35033f, -0.10408f, 0.29491f),
        make_float3(0.36219f, -0.09713f, 0.25894f), make_float3(0.37355f, -0.09144f, 0.22638f), make_float3(0.38451f, -0.08677f, 0.19721f),
        make_float3(0.39514f, -0.08296f, 0.17130f), make_float3(0.40551f, -0.07987f, 0.14843f), make_float3(0.41565f, -0.07736f, 0.12835f),
        make_float3(0.42562f, -0.07533f, 0.11079f), make_float3(0.43546f, -0.07371f, 0.09548f), make_float3(0.44519f, -0.07242f, 0.08219f),
        make_float3(0.45486f, -0.07138f, 0.07067f), make_float3(0.46449f, -0.07053f, 0.06071f), make_float3(0.47413f, -0.06981f, 0.05212f),
        make_float3(0.48380f, -0.06914f, 0.04472f), make_float3(0.49355f, -0.06841f, 0.03835f), make_float3(0.50342f, -0.06755f, 0.03289f),
        make_float3(0.51345f, -0.06643f, 0.02821f), make_float3(0.52370f, -0.06487f, 0.02421f), make_float3(0.53423f, -0.06265f, 0.02078f),
        make_float3(0.54512f, -0.05946f, 0.01786f), make_float3(0.55646f, -0.05481f, 0.01537f), make_float3(0.56840f, -0.04776f, 0.01325f),
        make_float3(0.58119f, -0.03618f, 0.01145f), make_float3(0.59545f, -0.00000f, 0.00995f),


        make_float3(0.22163f, -0.20055f, 0.63021f), make_float3(0.25410f, -0.20231f, 0.59968f), make_float3(0.28326f, -0.18906f, 0.56225f),
        make_float3(0.30475f, -0.17238f, 0.52492f), make_float3(0.32266f, -0.15662f, 0.48555f), make_float3(0.33852f, -0.14279f, 0.44484f),
        make_float3(0.35302f, -0.13097f, 0.40410f), make_float3(0.36656f, -0.12093f, 0.36452f), make_float3(0.37937f, -0.11245f, 0.32693f),
        make_float3(0.39160f, -0.10523f, 0.29188f), make_float3(0.40334f, -0.09909f, 0.25962f), make_float3(0.41470f, -0.09386f, 0.23022f),
        make_float3(0.42572f, -0.08939f, 0.20363f), make_float3(0.43646f, -0.08557f, 0.17973f), make_float3(0.44696f, -0.08227f, 0.15834f),
        make_float3(0.45725f, -0.07943f, 0.13927f), make_float3(0.46736f, -0.07697f, 0.12233f), make_float3(0.47733f, -0.07481f, 0.10732f),
        make_float3(0.48718f, -0.07290f, 0.09404f), make_float3(0.49693f, -0.07117f, 0.08234f), make_float3(0.50661f, -0.06954f, 0.07203f),
        make_float3(0.51626f, -0.06797f, 0.06298f), make_float3(0.52588f, -0.06635f, 0.05503f), make_float3(0.53552f, -0.06465f, 0.04807f),
        make_float3(0.54521f, -0.06270f, 0.04198f), make_float3(0.55498f, -0.06041f, 0.03667f), make_float3(0.56487f, -0.05757f, 0.03203f),
        make_float3(0.57494f, -0.05392f, 0.02799f), make_float3(0.58526f, -0.04906f, 0.02447f), make_float3(0.59592f, -0.04224f, 0.02142f),
        make_float3(0.60709f, -0.03161f, 0.01879f), make_float3(0.61917f, -0.00000f, 0.01653f),


        make_float3(0.24798f, -0.19716f, 0.63007f), make_float3(0.28069f, -0.19930f, 0.60332f), make_float3(0.31015f, -0.18883f, 0.56936f),
        make_float3(0.33172f, -0.17504f, 0.53486f), make_float3(0.34964f, -0.16162f, 0.49830f), make_float3(0.36550f, -0.14954f, 0.46037f),
        make_float3(0.38006f, -0.13890f, 0.42220f), make_float3(0.39370f, -0.12960f, 0.38484f), make_float3(0.40666f, -0.12148f, 0.34906f),
        make_float3(0.41907f, -0.11438f, 0.31535f), make_float3(0.43103f, -0.10815f, 0.28399f), make_float3(0.44261f, -0.10266f, 0.25508f),
        make_float3(0.45385f, -0.09783f, 0.22861f), make_float3(0.46481f, -0.09354f, 0.20450f), make_float3(0.47551f, -0.08974f, 0.18264f),
        make_float3(0.48599f, -0.08633f, 0.16289f), make_float3(0.49627f, -0.08326f, 0.14510f), make_float3(0.50638f, -0.08047f, 0.12910f),
        make_float3(0.51633f, -0.07791f, 0.11476f), make_float3(0.52614f, -0.07551f, 0.10192f), make_float3(0.53584f, -0.07321f, 0.09045f),
        make_float3(0.54546f, -0.07096f, 0.08022f), make_float3(0.55500f, -0.06867f, 0.07111f), make_float3(0.56449f, -0.06628f, 0.06301f),
        make_float3(0.57397f, -0.06367f, 0.05582f), make_float3(0.58345f, -0.06076f, 0.04944f), make_float3(0.59297f, -0.05731f, 0.04379f),
        make_float3(0.60257f, -0.05316f, 0.03879f), make_float3(0.61231f, -0.04788f, 0.03438f), make_float3(0.62225f, -0.04079f, 0.03050f),
        make_float3(0.63252f, -0.03022f, 0.02708f), make_float3(0.64336f, 0.00000f, 0.02411f),


        make_float3(0.27722f, -0.19199f, 0.63565f), make_float3(0.30981f, -0.19408f, 0.61130f), make_float3(0.33943f, -0.18579f, 0.57961f),
        make_float3(0.36102f, -0.17447f, 0.54708f), make_float3(0.37889f, -0.16322f, 0.51263f), make_float3(0.39470f, -0.15282f, 0.47688f),
        make_float3(0.40920f, -0.14347f, 0.44082f), make_float3(0.42282f, -0.13509f, 0.40537f), make_float3(0.43577f, -0.12760f, 0.37118f),
        make_float3(0.44819f, -0.12084f, 0.33873f), make_float3(0.46018f, -0.11477f, 0.30827f), make_float3(0.47180f, -0.10928f, 0.27992f),
        make_float3(0.48310f, -0.10430f, 0.25370f), make_float3(0.49411f, -0.09979f, 0.22957f), make_float3(0.50485f, -0.09566f, 0.20745f),
        make_float3(0.51535f, -0.09188f, 0.18722f), make_float3(0.52564f, -0.08837f, 0.16879f), make_float3(0.53572f, -0.08512f, 0.15202f),
        make_float3(0.54562f, -0.08205f, 0.13680f), make_float3(0.55536f, -0.07912f, 0.12300f), make_float3(0.56495f, -0.07630f, 0.11052f),
        make_float3(0.57441f, -0.07350f, 0.09925f), make_float3(0.58375f, -0.07067f, 0.08908f), make_float3(0.59300f, -0.06773f, 0.07992f),
        make_float3(0.60219f, -0.06459f, 0.07168f), make_float3(0.61132f, -0.06115f, 0.06428f), make_float3(0.62043f, -0.05724f, 0.05763f),
        make_float3(0.62954f, -0.05265f, 0.05168f), make_float3(0.63871f, -0.04703f, 0.04636f), make_float3(0.64797f, -0.03973f, 0.04161f),
        make_float3(0.65743f, -0.02917f, 0.03737f), make_float3(0.66725f, -0.00000f, 0.03363f),


        make_float3(0.30941f, -0.18550f, 0.64456f), make_float3(0.34168f, -0.18712f, 0.62252f), make_float3(0.37132f, -0.18024f, 0.59322f),
        make_float3(0.39288f, -0.17081f, 0.56272f), make_float3(0.41063f, -0.16130f, 0.53026f), make_float3(0.42627f, -0.15243f, 0.49646f),
        make_float3(0.44059f, -0.14431f, 0.46224f), make_float3(0.45403f, -0.13689f, 0.42843f), make_float3(0.46680f, -0.13010f, 0.39561f),
        make_float3(0.47906f, -0.12388f, 0.36425f), make_float3(0.49089f, -0.11815f, 0.33457f), make_float3(0.50236f, -0.11285f, 0.30671f),
        make_float3(0.51350f, -0.10796f, 0.28072f), make_float3(0.52435f, -0.10341f, 0.25657f), make_float3(0.53493f, -0.09917f, 0.23422f),
        make_float3(0.54526f, -0.09519f, 0.21359f), make_float3(0.55537f, -0.09146f, 0.19458f), make_float3(0.56525f, -0.08792f, 0.17712f),
        make_float3(0.57494f, -0.08453f, 0.16109f), make_float3(0.58444f, -0.08126f, 0.14642f), make_float3(0.59377f, -0.07806f, 0.13299f),
        make_float3(0.60294f, -0.07488f, 0.12073f), make_float3(0.61197f, -0.07164f, 0.10955f), make_float3(0.62087f, -0.06830f, 0.09937f),
        make_float3(0.62966f, -0.06479f, 0.09010f), make_float3(0.63836f, -0.06097f, 0.08168f), make_float3(0.64699f, -0.05673f, 0.07404f),
        make_float3(0.65557f, -0.05183f, 0.06711f), make_float3(0.66415f, -0.04599f, 0.06085f), make_float3(0.67274f, -0.03858f, 0.05518f),
        make_float3(0.68143f, -0.02812f, 0.05008f), make_float3(0.69032f, -0.00000f, 0.04550f),


        make_float3(0.34235f, -0.17826f, 0.65206f), make_float3(0.37374f, -0.17952f, 0.63152f), make_float3(0.40289f, -0.17391f, 0.60389f),
        make_float3(0.42408f, -0.16614f, 0.57496f), make_float3(0.44148f, -0.15827f, 0.54412f), make_float3(0.45676f, -0.15084f, 0.51197f),
        make_float3(0.47071f, -0.14393f, 0.47937f), make_float3(0.48377f, -0.13752f, 0.44704f), make_float3(0.49618f, -0.13155f, 0.41554f),
        make_float3(0.50808f, -0.12598f, 0.38525f), make_float3(0.51957f, -0.12076f, 0.35644f), make_float3(0.53070f, -0.11585f, 0.32922f),
        make_float3(0.54151f, -0.11120f, 0.30365f), make_float3(0.55204f, -0.10681f, 0.27973f), make_float3(0.56231f, -0.10265f, 0.25742f),
        make_float3(0.57233f, -0.09867f, 0.23667f), make_float3(0.58212f, -0.09486f, 0.21741f), make_float3(0.59168f, -0.09120f, 0.19957f),
        make_float3(0.60106f, -0.08765f, 0.18306f), make_float3(0.61023f, -0.08416f, 0.16782f), make_float3(0.61923f, -0.08072f, 0.15376f),
        make_float3(0.62806f, -0.07727f, 0.14081f), make_float3(0.63672f, -0.07376f, 0.12890f), make_float3(0.64525f, -0.07012f, 0.11795f),
        make_float3(0.65365f, -0.06628f, 0.10790f), make_float3(0.66194f, -0.06216f, 0.09869f), make_float3(0.67013f, -0.05760f, 0.09025f),
        make_float3(0.67825f, -0.05241f, 0.08254f), make_float3(0.68632f, -0.04628f, 0.07549f), make_float3(0.69436f, -0.03862f, 0.06906f),
        make_float3(0.70244f, -0.02799f, 0.06321f), make_float3(0.71061f, 0.00000f, 0.05791f),


        make_float3(0.37818f, -0.17017f, 0.65945f), make_float3(0.40866f, -0.17100f, 0.64009f), make_float3(0.43736f, -0.16627f, 0.61399f),
        make_float3(0.45826f, -0.15979f, 0.58661f), make_float3(0.47535f, -0.15320f, 0.55744f), make_float3(0.49027f, -0.14692f, 0.52704f),
        make_float3(0.50383f, -0.14103f, 0.49617f), make_float3(0.51647f, -0.13549f, 0.46545f), make_float3(0.52846f, -0.13026f, 0.43541f),
        make_float3(0.53992f, -0.12530f, 0.40641f), make_float3(0.55095f, -0.12057f, 0.37864f), make_float3(0.56162f, -0.11605f, 0.35226f),
        make_float3(0.57197f, -0.11171f, 0.32733f), make_float3(0.58204f, -0.10753f, 0.30384f), make_float3(0.59185f, -0.10350f, 0.28179f),
        make_float3(0.60141f, -0.09961f, 0.26113f), make_float3(0.61074f, -0.09583f, 0.24181f), make_float3(0.61985f, -0.09214f, 0.22378f),
        make_float3(0.62877f, -0.08853f, 0.20696f), make_float3(0.63747f, -0.08494f, 0.19131f), make_float3(0.64600f, -0.08136f, 0.17675f),
        make_float3(0.65435f, -0.07776f, 0.16324f), make_float3(0.66254f, -0.07407f, 0.15070f), make_float3(0.67057f, -0.07025f, 0.13908f),
        make_float3(0.67846f, -0.06622f, 0.12832f), make_float3(0.68622f, -0.06190f, 0.11837f), make_float3(0.69387f, -0.05716f, 0.10919f),
        make_float3(0.70143f, -0.05182f, 0.10071f), make_float3(0.70890f, -0.04557f, 0.09289f), make_float3(0.71632f, -0.03786f, 0.08570f),
        make_float3(0.72371f, -0.02730f, 0.07909f), make_float3(0.73113f, -0.00000f, 0.07304f),


        make_float3(0.41622f, -0.16117f, 0.66557f), make_float3(0.44565f, -0.16160f, 0.64703f), make_float3(0.47380f, -0.15756f, 0.62216f),
        make_float3(0.49434f, -0.15215f, 0.59618f), make_float3(0.51107f, -0.14662f, 0.56863f), make_float3(0.52562f, -0.14132f, 0.53998f),
        make_float3(0.53876f, -0.13629f, 0.51088f), make_float3(0.55092f, -0.13152f, 0.48188f), make_float3(0.56240f, -0.12695f, 0.45343f),
        make_float3(0.57332f, -0.12256f, 0.42585f), make_float3(0.58380f, -0.11831f, 0.39935f), make_float3(0.59391f, -0.11419f, 0.37404f),
        make_float3(0.60370f, -0.11017f, 0.34998f), make_float3(0.61319f, -0.10625f, 0.32718f), make_float3(0.62242f, -0.10243f, 0.30564f),
        make_float3(0.63141f, -0.09869f, 0.28533f), make_float3(0.64017f, -0.09501f, 0.26621f), make_float3(0.64871f, -0.09139f, 0.24823f),
        make_float3(0.65705f, -0.08778f, 0.23136f), make_float3(0.66519f, -0.08420f, 0.21553f), make_float3(0.67315f, -0.08059f, 0.20071f),
        make_float3(0.68092f, -0.07692f, 0.18683f), make_float3(0.68853f, -0.07317f, 0.17386f), make_float3(0.69599f, -0.06927f, 0.16174f),
        make_float3(0.70329f, -0.06516f, 0.15044f), make_float3(0.71046f, -0.06075f, 0.13990f), make_float3(0.71751f, -0.05595f, 0.13009f),
        make_float3(0.72444f, -0.05055f, 0.12095f), make_float3(0.73128f, -0.04430f, 0.11246f), make_float3(0.73804f, -0.03667f, 0.10458f),
        make_float3(0.74474f, -0.02633f, 0.09728f), make_float3(0.75140f, 0.00000f, 0.09052f),


        make_float3(0.45566f, -0.15078f, 0.67181f), make_float3(0.48376f, -0.15083f, 0.65404f), make_float3(0.51104f, -0.14734f, 0.63030f),
        make_float3(0.53106f, -0.14281f, 0.60566f), make_float3(0.54730f, -0.13818f, 0.57963f), make_float3(0.56134f, -0.13375f, 0.55265f),
        make_float3(0.57395f, -0.12949f, 0.52523f), make_float3(0.58556f, -0.12542f, 0.49788f), make_float3(0.59643f, -0.12148f, 0.47100f),
        make_float3(0.60673f, -0.11762f, 0.44485f), make_float3(0.61657f, -0.11387f, 0.41962f), make_float3(0.62603f, -0.11017f, 0.39542f),
        make_float3(0.63516f, -0.10653f, 0.37231f), make_float3(0.64399f, -0.10294f, 0.35030f), make_float3(0.65257f, -0.09939f, 0.32939f),
        make_float3(0.66090f, -0.09589f, 0.30956f), make_float3(0.66900f, -0.09240f, 0.29078f), make_float3(0.67689f, -0.08893f, 0.27303f),
        make_float3(0.68459f, -0.08546f, 0.25624f), make_float3(0.69209f, -0.08197f, 0.24041f), make_float3(0.69941f, -0.07844f, 0.22547f),
        make_float3(0.70656f, -0.07483f, 0.21139f), make_float3(0.71353f, -0.07112f, 0.19815f), make_float3(0.72036f, -0.06724f, 0.18568f),
        make_float3(0.72704f, -0.06317f, 0.17397f), make_float3(0.73357f, -0.05880f, 0.16298f), make_float3(0.73999f, -0.05404f, 0.15266f),
        make_float3(0.74629f, -0.04872f, 0.14298f), make_float3(0.75247f, -0.04259f, 0.13392f), make_float3(0.75857f, -0.03514f, 0.12544f),
        make_float3(0.76459f, -0.02515f, 0.11752f), make_float3(0.77054f, -0.00000f, 0.11012f),


        make_float3(0.49546f, -0.13868f, 0.67921f), make_float3(0.52189f, -0.13841f, 0.66231f), make_float3(0.54795f, -0.13540f, 0.63977f),
        make_float3(0.56715f, -0.13163f, 0.61642f), make_float3(0.58273f, -0.12780f, 0.59185f), make_float3(0.59613f, -0.12410f, 0.56640f),
        make_float3(0.60809f, -0.12056f, 0.54055f), make_float3(0.61905f, -0.11715f, 0.51472f), make_float3(0.62925f, -0.11379f, 0.48927f),
        make_float3(0.63886f, -0.11049f, 0.46444f), make_float3(0.64800f, -0.10724f, 0.44041f), make_float3(0.65676f, -0.10400f, 0.41727f),
        make_float3(0.66517f, -0.10079f, 0.39507f), make_float3(0.67330f, -0.09757f, 0.37385f), make_float3(0.68117f, -0.09438f, 0.35358f),
        make_float3(0.68879f, -0.09119f, 0.33427f), make_float3(0.69620f, -0.08798f, 0.31589f), make_float3(0.70340f, -0.08477f, 0.29841f),
        make_float3(0.71041f, -0.08153f, 0.28180f), make_float3(0.71724f, -0.07825f, 0.26604f), make_float3(0.72389f, -0.07490f, 0.25108f),
        make_float3(0.73037f, -0.07146f, 0.23691f), make_float3(0.73669f, -0.06790f, 0.22349f), make_float3(0.74288f, -0.06418f, 0.21078f),
        make_float3(0.74892f, -0.06025f, 0.19877f), make_float3(0.75481f, -0.05603f, 0.18742f), make_float3(0.76059f, -0.05144f, 0.17669f),
        make_float3(0.76625f, -0.04631f, 0.16657f), make_float3(0.77180f, -0.04042f, 0.15703f), make_float3(0.77725f, -0.03329f, 0.14803f),
        make_float3(0.78262f, -0.02378f, 0.13957f), make_float3(0.78790f, -0.00000f, 0.13161f),


        make_float3(0.53465f, -0.12522f, 0.68615f), make_float3(0.55925f, -0.12477f, 0.67008f), make_float3(0.58386f, -0.12219f, 0.64863f),
        make_float3(0.60211f, -0.11908f, 0.62649f), make_float3(0.61691f, -0.11598f, 0.60327f), make_float3(0.62960f, -0.11298f, 0.57927f),
        make_float3(0.64087f, -0.11008f, 0.55488f), make_float3(0.65112f, -0.10725f, 0.53052f), make_float3(0.66062f, -0.10446f, 0.50646f),
        make_float3(0.66953f, -0.10170f, 0.48294f), make_float3(0.67795f, -0.09893f, 0.46011f), make_float3(0.68598f, -0.09617f, 0.43805f),
        make_float3(0.69369f, -0.09339f, 0.41682f), make_float3(0.70109f, -0.09059f, 0.39644f), make_float3(0.70824f, -0.08778f, 0.37690f),
        make_float3(0.71515f, -0.08494f, 0.35821f), make_float3(0.72185f, -0.08209f, 0.34034f), make_float3(0.72836f, -0.07918f, 0.32326f),
        make_float3(0.73467f, -0.07624f, 0.30696f), make_float3(0.74082f, -0.07324f, 0.29141f), make_float3(0.74679f, -0.07016f, 0.27658f),
        make_float3(0.75261f, -0.06698f, 0.26246f), make_float3(0.75828f, -0.06367f, 0.24901f), make_float3(0.76381f, -0.06018f, 0.23621f),
        make_float3(0.76920f, -0.05650f, 0.22404f), make_float3(0.77446f, -0.05253f, 0.21247f), make_float3(0.77961f, -0.04820f, 0.20148f),
        make_float3(0.78464f, -0.04336f, 0.19105f), make_float3(0.78957f, -0.03781f, 0.18115f), make_float3(0.79440f, -0.03111f, 0.17176f),
        make_float3(0.79913f, -0.02219f, 0.16287f), make_float3(0.80378f, -0.00000f, 0.15446f),


        make_float3(0.57256f, -0.11055f, 0.69171f), make_float3(0.59531f, -0.11002f, 0.67633f), make_float3(0.61841f, -0.10792f, 0.65578f),
        make_float3(0.63563f, -0.10544f, 0.63470f), make_float3(0.64962f, -0.10298f, 0.61271f), make_float3(0.66156f, -0.10060f, 0.59006f),
        make_float3(0.67214f, -0.09828f, 0.56710f), make_float3(0.68170f, -0.09602f, 0.54415f), make_float3(0.69050f, -0.09375f, 0.52148f),
        make_float3(0.69871f, -0.09149f, 0.49928f), make_float3(0.70643f, -0.08920f, 0.47768f), make_float3(0.71376f, -0.08689f, 0.45676f),
        make_float3(0.72074f, -0.08455f, 0.43658f), make_float3(0.72745f, -0.08218f, 0.41714f), make_float3(0.73389f, -0.07976f, 0.39844f),
        make_float3(0.74011f, -0.07732f, 0.38048f), make_float3(0.74611f, -0.07484f, 0.36325f), make_float3(0.75193f, -0.07230f, 0.34672f),
        make_float3(0.75756f, -0.06969f, 0.33088f), make_float3(0.76304f, -0.06702f, 0.31571f), make_float3(0.76835f, -0.06427f, 0.30118f),
        make_float3(0.77352f, -0.06140f, 0.28727f), make_float3(0.77854f, -0.05841f, 0.27397f), make_float3(0.78343f, -0.05525f, 0.26124f),
        make_float3(0.78818f, -0.05188f, 0.24909f), make_float3(0.79283f, -0.04824f, 0.23747f), make_float3(0.79736f, -0.04426f, 0.22638f),
        make_float3(0.80178f, -0.03982f, 0.21580f), make_float3(0.80609f, -0.03471f, 0.20571f), make_float3(0.81032f, -0.02854f, 0.19608f),
        make_float3(0.81445f, -0.02034f, 0.18691f), make_float3(0.81849f, 0.00001f, 0.17818f),


        make_float3(0.60872f, -0.09464f, 0.69662f), make_float3(0.62966f, -0.09415f, 0.68177f), make_float3(0.65120f, -0.09250f, 0.66197f),
        make_float3(0.66737f, -0.09060f, 0.64182f), make_float3(0.68051f, -0.08873f, 0.62094f), make_float3(0.69172f, -0.08690f, 0.59955f),
        make_float3(0.70159f, -0.08513f, 0.57791f), make_float3(0.71047f, -0.08335f, 0.55632f), make_float3(0.71860f, -0.08158f, 0.53498f),
        make_float3(0.72612f, -0.07978f, 0.51406f), make_float3(0.73318f, -0.07795f, 0.49369f), make_float3(0.73983f, -0.07608f, 0.47393f),
        make_float3(0.74614f, -0.07417f, 0.45480f), make_float3(0.75217f, -0.07223f, 0.43634f), make_float3(0.75795f, -0.07023f, 0.41855f),
        make_float3(0.76350f, -0.06820f, 0.40140f), make_float3(0.76885f, -0.06611f, 0.38489f), make_float3(0.77401f, -0.06395f, 0.36901f),
        make_float3(0.77900f, -0.06175f, 0.35374f), make_float3(0.78383f, -0.05946f, 0.33905f), make_float3(0.78851f, -0.05707f, 0.32494f),
        make_float3(0.79306f, -0.05459f, 0.31137f), make_float3(0.79745f, -0.05197f, 0.29835f), make_float3(0.80173f, -0.04920f, 0.28584f),
        make_float3(0.80590f, -0.04623f, 0.27384f), make_float3(0.80993f, -0.04302f, 0.26232f), make_float3(0.81388f, -0.03949f, 0.25127f),
        make_float3(0.81771f, -0.03554f, 0.24068f), make_float3(0.82144f, -0.03097f, 0.23053f), make_float3(0.82508f, -0.02547f, 0.22081f),
        make_float3(0.82863f, -0.01815f, 0.21150f), make_float3(0.83210f, 0.00000f, 0.20258f),


        make_float3(0.64301f, -0.07753f, 0.70123f), make_float3(0.66223f, -0.07712f, 0.68683f), make_float3(0.68223f, -0.07593f, 0.66763f),
        make_float3(0.69736f, -0.07455f, 0.64827f), make_float3(0.70966f, -0.07320f, 0.62840f), make_float3(0.72015f, -0.07187f, 0.60815f),
        make_float3(0.72934f, -0.07056f, 0.58775f), make_float3(0.73757f, -0.06924f, 0.56743f), make_float3(0.74506f, -0.06791f, 0.54737f),
        make_float3(0.75196f, -0.06655f, 0.52771f), make_float3(0.75838f, -0.06515f, 0.50855f), make_float3(0.76441f, -0.06371f, 0.48992f),
        make_float3(0.77009f, -0.06222f, 0.47188f), make_float3(0.77550f, -0.06069f, 0.45443f), make_float3(0.78065f, -0.05912f, 0.43757f),
        make_float3(0.78559f, -0.05749f, 0.42129f), make_float3(0.79032f, -0.05583f, 0.40557f), make_float3(0.79488f, -0.05409f, 0.39042f),
        make_float3(0.79926f, -0.05229f, 0.37579f), make_float3(0.80349f, -0.05042f, 0.36169f), make_float3(0.80758f, -0.04846f, 0.34810f),
        make_float3(0.81154f, -0.04640f, 0.33499f), make_float3(0.81537f, -0.04423f, 0.32235f), make_float3(0.81906f, -0.04191f, 0.31018f),
        make_float3(0.82266f, -0.03941f, 0.29845f), make_float3(0.82614f, -0.03670f, 0.28716f), make_float3(0.82952f, -0.03372f, 0.27629f),
        make_float3(0.83279f, -0.03035f, 0.26582f), make_float3(0.83599f, -0.02647f, 0.25574f), make_float3(0.83908f, -0.02178f, 0.24605f),
        make_float3(0.84209f, -0.01551f, 0.23672f), make_float3(0.84502f, 0.00001f, 0.22776f),


        make_float3(0.67520f, -0.05931f, 0.70649f), make_float3(0.69276f, -0.05903f, 0.69240f), make_float3(0.71128f, -0.05822f, 0.67366f),
        make_float3(0.72537f, -0.05730f, 0.65499f), make_float3(0.73687f, -0.05639f, 0.63599f), make_float3(0.74664f, -0.05549f, 0.61679f),
        make_float3(0.75518f, -0.05459f, 0.59754f), make_float3(0.76280f, -0.05368f, 0.57840f), make_float3(0.76970f, -0.05276f, 0.55954f),
        make_float3(0.77600f, -0.05180f, 0.54106f), make_float3(0.78185f, -0.05080f, 0.52303f), make_float3(0.78730f, -0.04977f, 0.50551f),
        make_float3(0.79243f, -0.04869f, 0.48852f), make_float3(0.79726f, -0.04757f, 0.47207f), make_float3(0.80185f, -0.04641f, 0.45614f),
        make_float3(0.80622f, -0.04521f, 0.44074f), make_float3(0.81040f, -0.04396f, 0.42583f), make_float3(0.81441f, -0.04265f, 0.41143f),
        make_float3(0.81826f, -0.04130f, 0.39750f), make_float3(0.82195f, -0.03988f, 0.38404f), make_float3(0.82551f, -0.03837f, 0.37102f),
        make_float3(0.82892f, -0.03679f, 0.35844f), make_float3(0.83223f, -0.03510f, 0.34628f), make_float3(0.83541f, -0.03331f, 0.33452f),
        make_float3(0.83849f, -0.03135f, 0.32316f), make_float3(0.84146f, -0.02922f, 0.31218f), make_float3(0.84433f, -0.02686f, 0.30158f),
        make_float3(0.84711f, -0.02421f, 0.29134f), make_float3(0.84980f, -0.02113f, 0.28144f), make_float3(0.85241f, -0.01739f, 0.27189f),
        make_float3(0.85493f, -0.01240f, 0.26266f), make_float3(0.85737f, -0.00000f, 0.25376f),


        make_float3(0.70511f, -0.04013f, 0.71307f), make_float3(0.72112f, -0.03997f, 0.69919f), make_float3(0.73819f, -0.03950f, 0.68079f),
        make_float3(0.75128f, -0.03897f, 0.66265f), make_float3(0.76198f, -0.03845f, 0.64443f), make_float3(0.77108f, -0.03791f, 0.62615f),
        make_float3(0.77901f, -0.03737f, 0.60793f), make_float3(0.78606f, -0.03682f, 0.58989f), make_float3(0.79240f, -0.03625f, 0.57212f),
        make_float3(0.79819f, -0.03565f, 0.55474f), make_float3(0.80350f, -0.03502f, 0.53779f), make_float3(0.80844f, -0.03437f, 0.52130f),
        make_float3(0.81305f, -0.03367f, 0.50532f), make_float3(0.81738f, -0.03295f, 0.48981f), make_float3(0.82148f, -0.03220f, 0.47479f),
        make_float3(0.82537f, -0.03142f, 0.46024f), make_float3(0.82906f, -0.03059f, 0.44615f), make_float3(0.83259f, -0.02973f, 0.43250f),
        make_float3(0.83596f, -0.02882f, 0.41928f), make_float3(0.83918f, -0.02787f, 0.40648f), make_float3(0.84227f, -0.02686f, 0.39407f),
        make_float3(0.84523f, -0.02578f, 0.38205f), make_float3(0.84808f, -0.02463f, 0.37040f), make_float3(0.85081f, -0.02339f, 0.35913f),
        make_float3(0.85345f, -0.02205f, 0.34819f), make_float3(0.85597f, -0.02057f, 0.33761f), make_float3(0.85842f, -0.01893f, 0.32734f),
        make_float3(0.86076f, -0.01707f, 0.31740f), make_float3(0.86302f, -0.01492f, 0.30777f), make_float3(0.86519f, -0.01229f, 0.29845f),
        make_float3(0.86729f, -0.00878f, 0.28941f), make_float3(0.86932f, -0.00000f, 0.28066f),


        make_float3(0.73154f, -0.02030f, 0.72031f), make_float3(0.74609f, -0.02024f, 0.70643f), make_float3(0.76177f, -0.02006f, 0.68812f),
        make_float3(0.77389f, -0.01984f, 0.67034f), make_float3(0.78383f, -0.01961f, 0.65270f), make_float3(0.79228f, -0.01939f, 0.63519f),
        make_float3(0.79965f, -0.01914f, 0.61784f), make_float3(0.80618f, -0.01889f, 0.60072f), make_float3(0.81203f, -0.01864f, 0.58392f),
        make_float3(0.81735f, -0.01837f, 0.56751f), make_float3(0.82221f, -0.01807f, 0.55151f), make_float3(0.82672f, -0.01776f, 0.53597f),
        make_float3(0.83089f, -0.01744f, 0.52087f), make_float3(0.83482f, -0.01710f, 0.50623f), make_float3(0.83850f, -0.01673f, 0.49203f),
        make_float3(0.84198f, -0.01635f, 0.47828f), make_float3(0.84528f, -0.01595f, 0.46494f), make_float3(0.84842f, -0.01552f, 0.45201f),
        make_float3(0.85140f, -0.01508f, 0.43946f), make_float3(0.85424f, -0.01460f, 0.42730f), make_float3(0.85696f, -0.01408f, 0.41549f),
        make_float3(0.85955f, -0.01354f, 0.40404f), make_float3(0.86204f, -0.01296f, 0.39292f), make_float3(0.86442f, -0.01233f, 0.38212f),
        make_float3(0.86669f, -0.01164f, 0.37165f), make_float3(0.86887f, -0.01087f, 0.36148f), make_float3(0.87097f, -0.01002f, 0.35161f),
        make_float3(0.87297f, -0.00905f, 0.34203f), make_float3(0.87490f, -0.00791f, 0.33272f), make_float3(0.87673f, -0.00653f, 0.32369f),
        make_float3(0.87850f, -0.00466f, 0.31492f), make_float3(0.88020f, 0.00001f, 0.30640f),


        make_float3(0.75486f, -0.00000f, 0.72806f), make_float3(0.76807f, 0.00000f, 0.71395f), make_float3(0.78246f, -0.00000f, 0.69552f),
        make_float3(0.79366f, -0.00000f, 0.67790f), make_float3(0.80290f, 0.00001f, 0.66069f), make_float3(0.81077f, 0.00001f, 0.64378f),
        make_float3(0.81763f, -0.00000f, 0.62716f), make_float3(0.82368f, -0.00000f, 0.61086f), make_float3(0.82912f, -0.00000f, 0.59491f),
        make_float3(0.83404f, -0.00000f, 0.57936f), make_float3(0.83852f, -0.00000f, 0.56423f), make_float3(0.84266f, 0.00000f, 0.54953f),
        make_float3(0.84649f, 0.00000f, 0.53526f), make_float3(0.85008f, -0.00000f, 0.52142f), make_float3(0.85343f, 0.00000f, 0.50800f),
        make_float3(0.85660f, -0.00000f, 0.49498f), make_float3(0.85959f, 0.00000f, 0.48235f), make_float3(0.86241f, -0.00000f, 0.47011f),
        make_float3(0.86510f, 0.00001f, 0.45821f), make_float3(0.86766f, 0.00000f, 0.44666f), make_float3(0.87010f, -0.00001f, 0.43545f),
        make_float3(0.87242f, 0.00000f, 0.42456f), make_float3(0.87464f, -0.00000f, 0.41398f), make_float3(0.87675f, 0.00000f, 0.40369f),
        make_float3(0.87877f, 0.00000f, 0.39370f), make_float3(0.88070f, -0.00000f, 0.38398f), make_float3(0.88255f, -0.00000f, 0.37453f),
        make_float3(0.88431f, 0.00000f, 0.36535f), make_float3(0.88600f, 0.00000f, 0.35642f), make_float3(0.88761f, -0.00000f, 0.34773f),
        make_float3(0.88915f, -0.00000f, 0.33929f), make_float3(0.89063f, -0.00000f, 0.33107f)
};

#endif // __KERNELCC__

#endif