#ifndef BVH_TESTS_H
#define BVH_TESTS_H

#include "ray.h"

#include <vector>

std::vector<Ray> create_ray_inter_vector()
{
    std::vector<Ray> rays;

    rays.push_back(Ray(Point(-0.035025f, 0.970885f, -0.066203f), Vector(0.131333f, -0.628322f, 0.766787f)));
    rays.push_back(Ray(Point(-0.035485f, 0.983855f, -0.066058f), Vector(0.950127f, -0.078177f, 0.301906f)));
    rays.push_back(Ray(Point(-0.035578f, 0.988599f, -0.066028f), Vector(0.986444f, 0.132046f, -0.097429f)));
    rays.push_back(Ray(Point(-0.036262f, 0.990519f, -0.065812f), Vector(-0.456679f, 0.689698f, 0.561927f)));
    rays.push_back(Ray(Point(-0.037448f, 1.031054f, -0.065437f), Vector(0.053811f, 0.920864f, 0.386152f)));
    rays.push_back(Ray(Point(-0.036065f, 1.001313f, -0.065874f), Vector(-0.724915f, -0.643387f, 0.246072f)));
    rays.push_back(Ray(Point(-0.036759f, 1.006047f, -0.065655f), Vector(-0.897585f, -0.026309f, 0.440056f)));
    rays.push_back(Ray(Point(-0.036554f, 1.01043f, -0.06572f), Vector(0.771941f, 0.176895f, 0.610586f)));
    rays.push_back(Ray(Point(-0.037274f, 1.025391f, -0.065493f), Vector(0.794456f, 0.577108f, 0.189173f)));
    rays.push_back(Ray(Point(-0.037612f, 1.027336f, -0.065386f), Vector(0.852981f, 0.097164f, 0.512818f)));
    rays.push_back(Ray(Point(0.596629f, 0.0001f, -0.212552f), Vector(0.620855f, 0.454949f, -0.638405f)));
    rays.push_back(Ray(Point(0.042498f, 0.6001f, 0.386415f), Vector(0.691927f, 0.651476f, 0.311154f)));
    rays.push_back(Ray(Point(-0.376678f, 0.0001f, 0.280161f), Vector(-0.264457f, 0.009405f, -0.964352f)));
    rays.push_back(Ray(Point(0.619748f, 0.0001f, -0.26434f), Vector(0.896164f, 0.321067f, -0.306277f)));
    rays.push_back(Ray(Point(0.9999f, 0.898655f, 0.262971f), Vector(-0.570861f, -0.691226f, 0.443085f)));
    rays.push_back(Ray(Point(0.9999f, 1.127223f, -0.16831f), Vector(-0.528733f, -0.332853f, -0.780801f)));
    rays.push_back(Ray(Point(0.018589f, 1.9899f, 0.336684f), Vector(0.419441f, -0.383443f, -0.822825f)));
    rays.push_back(Ray(Point(0.020369f, 0.0001f, 0.091317f), Vector(0.125118f, 0.202665f, -0.971222f)));
    rays.push_back(Ray(Point(0.720363f, 1.348355f, -1.0399f), Vector(0.925843f, -0.073668f, 0.370659f)));
    rays.push_back(Ray(Point(0.166112f, 0.236172f, -1.0399f), Vector(-0.782928f, 0.464641f, 0.413681f)));
    rays.push_back(Ray(Point(0.9999f, 0.295682f, -0.627325f), Vector(-0.029682f, 0.084511f, -0.99598f)));
    rays.push_back(Ray(Point(0.9999f, 0.136332f, -0.394297f), Vector(-0.512592f, 0.313232f, 0.799459f)));
    rays.push_back(Ray(Point(0.409619f, 0.755624f, -1.0399f), Vector(0.987451f, 0.043367f, 0.151852f)));
    rays.push_back(Ray(Point(-1.004869f, 0.141379f, 0.26302f), Vector(0.725977f, 0.188291f, -0.66144f)));
    rays.push_back(Ray(Point(-1.014811f, 0.977377f, 0.413901f), Vector(0.434342f, 0.873371f, 0.220387f)));
    rays.push_back(Ray(Point(-0.067955f, 0.0001f, 0.010397f), Vector(-0.573775f, 0.759307f, 0.306978f)));
    rays.push_back(Ray(Point(-0.377942f, 0.0001f, 0.168344f), Vector(0.677894f, 0.238412f, 0.695428f)));
    rays.push_back(Ray(Point(0.957577f, 1.9899f, -0.00025f), Vector(-0.960279f, -0.009986f, -0.278863f)));
    rays.push_back(Ray(Point(-0.033764f, 1.03253f, -0.066601f), Vector(-0.230912f, 0.794757f, 0.561284f)));
    rays.push_back(Ray(Point(-0.033662f, 1.037428f, -0.066633f), Vector(0.896316f, 0.432476f, -0.097883f)));
    rays.push_back(Ray(Point(-0.029927f, 1.035633f, -0.067813f), Vector(0.418508f, -0.56804f, 0.708648f)));
    rays.push_back(Ray(Point(-0.009636f, 1.03136f, -0.07422f), Vector(-0.662699f, 0.661189f, 0.351654f)));
    rays.push_back(Ray(Point(-0.025444f, 1.034688f, -0.069228f), Vector(0.670403f, 0.741945f, 0.008818f)));
    rays.push_back(Ray(Point(-0.025811f, 1.03875f, -0.069112f), Vector(-0.878856f, -0.115331f, 0.462937f)));
    rays.push_back(Ray(Point(-0.021776f, 1.034142f, -0.070387f), Vector(0.989849f, 0.0424f, 0.135651f)));
    rays.push_back(Ray(Point(-0.021551f, 1.037986f, -0.070458f), Vector(0.716508f, 0.543883f, 0.436815f)));
    rays.push_back(Ray(Point(-0.009529f, 1.03826f, -0.074254f), Vector(0.743249f, -0.555468f, 0.372875f)));
    rays.push_back(Ray(Point(-0.005484f, 1.034931f, -0.075531f), Vector(0.966518f, 0.143438f, -0.212764f)));
    rays.push_back(Ray(Point(-0.022762f, 0.965935f, -0.070075f), Vector(0.479333f, -0.705087f, 0.522582f)));
    rays.push_back(Ray(Point(-0.023495f, 0.97206f, -0.069844f), Vector(0.046675f, 0.807517f, 0.587995f)));
    rays.push_back(Ray(Point(-0.023313f, 0.980653f, -0.069901f), Vector(0.839923f, -0.513158f, -0.176631f)));
    rays.push_back(Ray(Point(-0.023956f, 0.982134f, -0.069698f), Vector(0.625252f, 0.757786f, -0.186601f)));
    rays.push_back(Ray(Point(-0.023768f, 0.993067f, -0.069758f), Vector(-0.810569f, -0.00494f, 0.585623f)));
    rays.push_back(Ray(Point(-0.024436f, 0.996338f, -0.069547f), Vector(0.810139f, 0.197453f, 0.551984f)));
    rays.push_back(Ray(Point(-0.019331f, 0.96819f, -0.071159f), Vector(-0.699973f, -0.570944f, 0.429023f)));
    rays.push_back(Ray(Point(-0.019432f, 0.97014f, -0.071127f), Vector(-0.367723f, 0.769677f, 0.521897f)));
    rays.push_back(Ray(Point(-0.019136f, 0.973864f, -0.07122f), Vector(0.939838f, 0.220378f, 0.261031f)));
    rays.push_back(Ray(Point(-0.019804f, 0.980675f, -0.071009f), Vector(0.796468f, -0.320303f, 0.512879f)));
    rays.push_back(Ray(Point(-0.03099f, 0.972558f, -0.067477f), Vector(0.374431f, 0.927091f, 0.01741f)));
    rays.push_back(Ray(Point(-0.020266f, 0.992862f, -0.070863f), Vector(0.81176f, -0.357997f, 0.461394f)));
    rays.push_back(Ray(Point(-0.031611f, 0.975447f, -0.067281f), Vector(-0.704296f, 0.654906f, 0.273981f)));
    rays.push_back(Ray(Point(-0.032271f, 0.99317f, -0.067073f), Vector(-0.573873f, 0.611895f, 0.544292f)));
    rays.push_back(Ray(Point(-0.031916f, 0.996303f, -0.067185f), Vector(0.767855f, 0.440459f, 0.465182f)));
    rays.push_back(Ray(Point(-0.026835f, 0.968048f, -0.068789f), Vector(0.901149f, -0.394942f, 0.178748f)));
    rays.push_back(Ray(Point(-0.02746f, 0.969872f, -0.068592f), Vector(0.18714f, -0.948636f, 0.255085f)));
    rays.push_back(Ray(Point(-0.02795f, 0.983984f, -0.068437f), Vector(-0.414718f, 0.733261f, 0.538829f)));
    rays.push_back(Ray(Point(-0.02778f, 0.991706f, -0.068491f), Vector(0.790064f, -0.517996f, 0.32784f)));
    rays.push_back(Ray(Point(-0.014903f, 0.971686f, -0.072557f), Vector(0.96348f, -0.040188f, 0.264747f)));
    rays.push_back(Ray(Point(-0.015745f, 0.978758f, -0.072291f), Vector(0.940414f, 0.33926f, 0.022889f)));
    rays.push_back(Ray(Point(-0.015398f, 0.983187f, -0.072401f), Vector(0.644952f, 0.707206f, 0.289649f)));
    rays.push_back(Ray(Point(-0.016235f, 0.991271f, -0.072136f), Vector(0.49205f, 0.674685f, 0.55017f)));
    rays.push_back(Ray(Point(-0.016402f, 0.994931f, -0.072084f), Vector(0.406496f, -0.602316f, 0.687005f)));
    rays.push_back(Ray(Point(-0.010943f, 0.96938f, -0.073807f), Vector(0.336267f, 0.935728f, 0.106476f)));
    rays.push_back(Ray(Point(-0.011737f, 0.980693f, -0.073557f), Vector(0.94081f, -0.293581f, -0.169375f)));
    rays.push_back(Ray(Point(-0.011382f, 0.985446f, -0.073669f), Vector(-0.717061f, 0.146049f, 0.681538f)));
    rays.push_back(Ray(Point(-0.012078f, 0.987887f, -0.07345f), Vector(0.726343f, -0.671343f, -0.147393f)));
    rays.push_back(Ray(Point(-0.01224f, 0.993404f, -0.073398f), Vector(-0.769911f, 0.325385f, 0.548964f)));
    rays.push_back(Ray(Point(-0.011873f, 0.99713f, -0.073514f), Vector(-0.058197f, 0.957212f, 0.283475f)));
    rays.push_back(Ray(Point(-0.017512f, 1.033312f, -0.071733f), Vector(0.096421f, -0.626343f, 0.773562f)));
    rays.push_back(Ray(Point(-0.007572f, 0.974982f, -0.074872f), Vector(0.80744f, 0.589641f, -0.019097f)));
    rays.push_back(Ray(Point(-0.007231f, 0.978515f, -0.07498f), Vector(0.663598f, -0.549897f, 0.507199f)));
    rays.push_back(Ray(Point(-0.007852f, 0.994711f, -0.074784f), Vector(0.499131f, 0.866523f, 0.002491f)));
    rays.push_back(Ray(Point(-0.003406f, 0.973253f, -0.076188f), Vector(0.813875f, -0.543774f, 0.204736f)));
    rays.push_back(Ray(Point(-0.003564f, 0.977111f, -0.076138f), Vector(0.802291f, 0.548323f, -0.235948f)));
    rays.push_back(Ray(Point(-0.003203f, 0.98071f, -0.076252f), Vector(0.035149f, 0.970269f, 0.239465f)));
    rays.push_back(Ray(Point(-0.003672f, 0.992025f, -0.076104f), Vector(-0.697105f, -0.401053f, 0.594308f)));
    rays.push_back(Ray(Point(-0.031171f, 0.962017f, -0.06742f), Vector(-0.254755f, 0.806199f, 0.533988f)));
    rays.push_back(Ray(Point(-0.027117f, 0.963174f, -0.0687f), Vector(0.654603f, -0.663525f, 0.362256f)));
    rays.push_back(Ray(Point(-0.01862f, 0.96369f, -0.071383f), Vector(0.386264f, 0.899113f, 0.205901f)));
    rays.push_back(Ray(Point(-0.015168f, 0.962413f, -0.072473f), Vector(0.604376f, 0.754123f, 0.256958f)));
    rays.push_back(Ray(Point(-0.011137f, 0.964215f, -0.073746f), Vector(0.978744f, -0.131253f, 0.157582f)));
    rays.push_back(Ray(Point(0.648056f, 0.351353f, 0.154403f), Vector(0.499447f, 0.19132f, -0.844955f)));
    rays.push_back(Ray(Point(0.9999f, 0.781552f, -0.94911f), Vector(-0.079991f, 0.823489f, 0.561665f)));
    rays.push_back(Ray(Point(0.9999f, 1.326105f, -0.927948f), Vector(-0.382255f, -0.432439f, -0.816625f)));
    rays.push_back(Ray(Point(-0.032604f, 1.001239f, -0.066968f), Vector(-0.046277f, 0.837645f, 0.544252f)));
    rays.push_back(Ray(Point(-0.03274f, 1.002866f, -0.066924f), Vector(0.819652f, 0.18723f, 0.541401f)));
    rays.push_back(Ray(Point(-0.032402f, 1.006779f, -0.067031f), Vector(0.31605f, -0.835707f, 0.449117f)));
    rays.push_back(Ray(Point(-0.033046f, 1.01296f, -0.066828f), Vector(0.919315f, 0.260455f, 0.294996f)));
    rays.push_back(Ray(Point(-0.032889f, 1.016872f, -0.066877f), Vector(-0.870212f, -0.230217f, 0.435582f)));
    rays.push_back(Ray(Point(-0.033239f, 1.019525f, -0.066767f), Vector(0.958862f, -0.073805f, -0.274112f)));
    rays.push_back(Ray(Point(-0.028588f, 0.999908f, -0.068236f), Vector(-0.690986f, -0.296141f, 0.659423f)));
    rays.push_back(Ray(Point(-0.028238f, 1.003759f, -0.068346f), Vector(-0.846728f, 0.429528f, 0.313938f)));
    rays.push_back(Ray(Point(-0.028391f, 1.008658f, -0.068298f), Vector(-0.278621f, 0.951491f, 0.130521f)));
    rays.push_back(Ray(Point(-0.028869f, 1.011449f, -0.068147f), Vector(0.733309f, 0.561731f, 0.383035f)));
    rays.push_back(Ray(Point(-0.029209f, 1.014778f, -0.068039f), Vector(0.805533f, 0.560511f, 0.19221f)));
    rays.push_back(Ray(Point(-0.029273f, 1.028976f, -0.068019f), Vector(-0.883469f, 0.02097f, 0.46802f)));
    rays.push_back(Ray(Point(-0.01645f, 1.008026f, -0.072069f), Vector(0.966864f, -0.149916f, 0.206636f)));
    rays.push_back(Ray(Point(-0.017176f, 1.014575f, -0.071839f), Vector(0.72823f, -0.517261f, 0.44958f)));
    rays.push_back(Ray(Point(-0.017018f, 1.020282f, -0.07189f), Vector(-0.768419f, 0.437339f, 0.467191f)));
    rays.push_back(Ray(Point(-0.017356f, 1.023684f, -0.071783f), Vector(-0.886585f, 0.257922f, 0.383984f)));
    rays.push_back(Ray(Point(-0.017202f, 1.028373f, -0.071831f), Vector(-0.649971f, -0.351848f, 0.673603f)));
    rays.push_back(Ray(Point(-0.012756f, 1.006801f, -0.073235f), Vector(-0.695116f, -0.050397f, 0.717129f)));
    rays.push_back(Ray(Point(-0.024236f, 1.005445f, -0.06961f), Vector(0.384879f, -0.745364f, 0.544335f)));
    rays.push_back(Ray(Point(-0.02483f, 1.006633f, -0.069422f), Vector(0.78597f, -0.616606f, -0.045248f)));
    rays.push_back(Ray(Point(-0.024881f, 1.022572f, -0.069406f), Vector(0.940457f, 0.202168f, -0.273256f)));
    rays.push_back(Ray(Point(-0.025245f, 1.026348f, -0.069291f), Vector(0.207838f, -0.712435f, 0.670254f)));
    rays.push_back(Ray(Point(-0.025598f, 1.030645f, -0.06918f), Vector(-0.036101f, 0.989838f, 0.137542f)));
    rays.push_back(Ray(Point(-0.020063f, 0.998152f, -0.070928f), Vector(0.888551f, 0.260385f, 0.377726f)));
    rays.push_back(Ray(Point(-0.020642f, 1.009218f, -0.070745f), Vector(0.693596f, 0.720262f, -0.012133f)));
    rays.push_back(Ray(Point(-0.021004f, 1.012558f, -0.070631f), Vector(0.740846f, 0.318337f, 0.591446f)));
    rays.push_back(Ray(Point(-0.020846f, 1.01718f, -0.07068f), Vector(0.722313f, -0.608329f, 0.328936f)));
    rays.push_back(Ray(Point(-0.021216f, 1.019512f, -0.070564f), Vector(0.504805f, 0.850298f, -0.148878f)));
    rays.push_back(Ray(Point(-0.021051f, 1.023621f, -0.070616f), Vector(0.114202f, -0.699571f, 0.705378f)));
    rays.push_back(Ray(Point(-0.021395f, 1.027471f, -0.070507f), Vector(0.580343f, -0.58154f, 0.570099f)));
    rays.push_back(Ray(Point(-0.008379f, 1.001043f, -0.074617f), Vector(0.455799f, -0.447967f, 0.769137f)));
    rays.push_back(Ray(Point(-0.008581f, 1.008362f, -0.074554f), Vector(0.770902f, 0.37364f, 0.515852f)));
    rays.push_back(Ray(Point(-0.008906f, 1.013951f, -0.074451f), Vector(-0.266494f, 0.888061f, 0.374604f)));
    rays.push_back(Ray(Point(-0.008747f, 1.018257f, -0.074501f), Vector(-0.032467f, 0.685401f, 0.727442f)));
    rays.push_back(Ray(Point(-0.009106f, 1.022561f, -0.074388f), Vector(0.511109f, -0.587361f, 0.627515f)));
    rays.push_back(Ray(Point(-0.009341f, 1.027055f, -0.074313f), Vector(0.508675f, 0.860957f, 0.001986f)));
    rays.push_back(Ray(Point(-0.00436f, 1.007939f, -0.075887f), Vector(0.591593f, 0.724392f, 0.353941f)));
    rays.push_back(Ray(Point(-0.005053f, 1.015392f, -0.075668f), Vector(-0.460402f, 0.756802f, 0.463983f)));
    rays.push_back(Ray(Point(-0.00489f, 1.019507f, -0.075719f), Vector(0.857249f, 0.514873f, 0.005452f)));
    rays.push_back(Ray(Point(-0.005583f, 1.03056f, -0.0755f), Vector(0.97587f, -0.197254f, 0.093643f)));
    rays.push_back(Ray(Point(-0.311951f, 1.9899f, 0.609596f), Vector(0.519254f, -0.815239f, -0.256437f)));
    rays.push_back(Ray(Point(0.291028f, 0.6001f, 0.475652f), Vector(-0.188486f, 0.963841f, -0.188369f)));
    rays.push_back(Ray(Point(0.837753f, 1.9899f, -0.057874f), Vector(-0.834881f, -0.53294f, 0.137658f)));
    rays.push_back(Ray(Point(-1.014468f, 0.908997f, 0.451714f), Vector(0.851033f, -0.523368f, 0.042762f)));
    rays.push_back(Ray(Point(0.9999f, 1.07791f, 0.069639f), Vector(-0.937721f, 0.275308f, -0.211862f)));
    rays.push_back(Ray(Point(-0.970465f, 1.9899f, 0.435633f), Vector(-0.735383f, -0.621747f, -0.269522f)));
    rays.push_back(Ray(Point(0.57689f, 0.6001f, 0.219942f), Vector(-0.323883f, 0.874273f, 0.361588f)));
    rays.push_back(Ray(Point(-0.220392f, 0.0001f, 0.929968f), Vector(0.702487f, 0.418345f, -0.57576f)));
    rays.push_back(Ray(Point(-0.334624f, 0.0001f, 0.365866f), Vector(0.759502f, 0.445723f, -0.473801f)));
    rays.push_back(Ray(Point(0.902787f, 1.9899f, 0.339958f), Vector(0.34488f, -0.171866f, 0.922778f)));
    rays.push_back(Ray(Point(0.71214f, 1.9899f, 0.742275f), Vector(-0.97136f, -0.187526f, -0.145923f)));
    rays.push_back(Ray(Point(0.89478f, 0.0001f, 0.029643f), Vector(0.903531f, 0.02516f, -0.427784f)));
    rays.push_back(Ray(Point(0.950969f, 0.0001f, -0.01465f), Vector(-0.016731f, 0.67685f, -0.73593f)));
    rays.push_back(Ray(Point(0.56545f, 1.9899f, -0.071923f), Vector(0.253221f, -0.958544f, 0.130662f)));
    rays.push_back(Ray(Point(0.9999f, 0.302848f, 0.176226f), Vector(-0.09551f, 0.993113f, -0.067848f)));
    rays.push_back(Ray(Point(0.9999f, 1.662994f, -0.371278f), Vector(-0.379218f, 0.596623f, -0.707272f)));
    rays.push_back(Ray(Point(0.03336f, 1.9899f, 0.172843f), Vector(-0.837501f, -0.235068f, 0.493291f)));
    rays.push_back(Ray(Point(0.250132f, 0.6001f, 0.378377f), Vector(0.735864f, 0.511887f, -0.443257f)));
    rays.push_back(Ray(Point(0.35583f, 1.9899f, 0.042329f), Vector(-0.876508f, -0.282339f, 0.389895f)));
    rays.push_back(Ray(Point(0.528922f, 0.0001f, -0.237989f), Vector(-0.446867f, 0.87468f, -0.187736f)));
    rays.push_back(Ray(Point(0.9999f, 0.664979f, -0.2557f), Vector(-0.952306f, 0.271556f, -0.139177f)));
    rays.push_back(Ray(Point(-1.01588f, 1.190061f, 0.881161f), Vector(0.043441f, 0.621817f, -0.781957f)));
    rays.push_back(Ray(Point(-1.017026f, 1.418095f, 0.643108f), Vector(0.545898f, 0.451314f, -0.705911f)));
    rays.push_back(Ray(Point(-0.072237f, 1.9899f, 0.220521f), Vector(0.498107f, -0.805589f, 0.320804f)));
    rays.push_back(Ray(Point(0.049192f, 0.6001f, 0.463426f), Vector(0.200074f, 0.978696f, 0.046083f)));
    rays.push_back(Ray(Point(-0.908544f, 0.0001f, 0.926819f), Vector(-0.390934f, 0.920022f, 0.027017f)));
    rays.push_back(Ray(Point(-0.43835f, 0.0001f, 0.712549f), Vector(0.343916f, 0.93894f, 0.01062f)));
    rays.push_back(Ray(Point(0.9999f, 0.355452f, -0.285097f), Vector(-0.736028f, -0.029218f, -0.676321f)));
    rays.push_back(Ray(Point(0.807638f, 1.9899f, -0.31788f), Vector(0.653151f, -0.448679f, 0.609985f)));
    rays.push_back(Ray(Point(-1.01486f, 0.987026f, 0.646362f), Vector(0.14508f, 0.972556f, -0.181895f)));
    rays.push_back(Ray(Point(0.9999f, 1.212836f, 0.211835f), Vector(-0.400935f, 0.71415f, -0.573795f)));
    rays.push_back(Ray(Point(0.9999f, 0.570557f, 0.585686f), Vector(-0.941648f, 0.267053f, -0.204897f)));
    rays.push_back(Ray(Point(-0.673886f, 0.0001f, 0.665641f), Vector(-0.242578f, 0.968026f, -0.063892f)));
    rays.push_back(Ray(Point(0.379932f, 1.9899f, -0.04837f), Vector(0.024022f, -0.919476f, 0.392412f)));
    rays.push_back(Ray(Point(-0.861845f, 0.0001f, 0.692297f), Vector(-0.481243f, 0.804202f, -0.348806f)));
    rays.push_back(Ray(Point(-0.967159f, 1.9899f, 0.819625f), Vector(0.802444f, -0.585867f, -0.113332f)));
    rays.push_back(Ray(Point(0.9999f, 1.588232f, 0.557971f), Vector(-0.328743f, 0.814631f, 0.477812f)));
    rays.push_back(Ray(Point(-0.155319f, 0.0001f, 0.540292f), Vector(0.633144f, 0.770925f, -0.069299f)));
    rays.push_back(Ray(Point(0.872396f, 0.0001f, 0.115687f), Vector(-0.702378f, 0.153103f, -0.695143f)));
    rays.push_back(Ray(Point(0.569661f, 0.6001f, 0.17942f), Vector(-0.983142f, 0.036153f, 0.179234f)));
    rays.push_back(Ray(Point(-0.007069f, 0.96229f, -0.075031f), Vector(0.606224f, -0.668053f, 0.431507f)));
    rays.push_back(Ray(Point(-0.002584f, 0.964736f, -0.076447f), Vector(0.839006f, 0.32606f, 0.435606f)));
    rays.push_back(Ray(Point(-0.511216f, 1.9899f, 0.669427f), Vector(0.43952f, -0.196467f, -0.876483f)));
    rays.push_back(Ray(Point(-1.016207f, 1.255106f, 0.517779f), Vector(0.954404f, -0.190206f, -0.230075f)));
    rays.push_back(Ray(Point(0.422289f, 1.9899f, 0.163647f), Vector(0.724707f, -0.087448f, 0.683485f)));
    rays.push_back(Ray(Point(0.9999f, 0.828617f, 0.089051f), Vector(-0.952993f, -0.197656f, -0.229645f)));
    rays.push_back(Ray(Point(-1.013959f, 0.80781f, 0.247459f), Vector(0.015732f, -0.32527f, -0.94549f)));
    rays.push_back(Ray(Point(-0.02839f, 0.302521f, 0.501236f), Vector(-0.529721f, 0.821764f, -0.21f)));
    rays.push_back(Ray(Point(-1.018294f, 1.670467f, -0.38636f), Vector(0.589257f, 0.698199f, 0.406563f)));
    rays.push_back(Ray(Point(-1.01969f, 1.948197f, 0.417555f), Vector(0.260844f, 0.950022f, 0.171519f)));
    rays.push_back(Ray(Point(0.144948f, 0.217685f, 0.630606f), Vector(-0.095618f, 0.987824f, 0.12272f)));
    rays.push_back(Ray(Point(0.551352f, 0.0001f, -0.699806f), Vector(-0.321793f, 0.796746f, 0.511513f)));
    rays.push_back(Ray(Point(0.9999f, 0.00303f, -0.020174f), Vector(-0.22066f, 0.972873f, -0.069485f)));
    rays.push_back(Ray(Point(0.927658f, 0.943135f, -1.0399f), Vector(-0.929089f, 0.347627f, 0.126285f)));
    rays.push_back(Ray(Point(0.9999f, 0.344955f, 0.152304f), Vector(-0.215792f, 0.890124f, 0.40139f)));
    rays.push_back(Ray(Point(0.837642f, 1.9899f, 0.060962f), Vector(0.549409f, -0.825884f, 0.126745f)));
    rays.push_back(Ray(Point(0.792053f, 1.9899f, -0.75893f), Vector(0.7913f, -0.45999f, -0.402807f)));
    rays.push_back(Ray(Point(0.900629f, 0.0001f, -0.360178f), Vector(0.810284f, 0.569893f, 0.13661f)));
    rays.push_back(Ray(Point(-1.018221f, 1.655825f, 0.482316f), Vector(0.150608f, 0.966351f, 0.208525f)));
    rays.push_back(Ray(Point(-1.017676f, 1.547437f, 0.653346f), Vector(0.936943f, -0.321794f, -0.136333f)));
    rays.push_back(Ray(Point(-0.487702f, 1.9899f, -0.66509f), Vector(-0.801209f, -0.580734f, 0.144266f)));
    rays.push_back(Ray(Point(-0.059378f, 0.967066f, -0.410554f), Vector(0.51953f, 0.297907f, -0.800837f)));
    rays.push_back(Ray(Point(-0.959996f, 1.9899f, -0.124791f), Vector(-0.782653f, -0.622246f, -0.016288f)));
    rays.push_back(Ray(Point(0.948069f, 0.0001f, -0.276074f), Vector(0.563083f, 0.3149f, 0.764052f)));
    rays.push_back(Ray(Point(-0.325264f, 1.9899f, -0.251423f), Vector(-0.471289f, -0.116694f, -0.874225f)));
    rays.push_back(Ray(Point(-0.142085f, 0.0001f, 0.060837f), Vector(-0.851901f, 0.441932f, 0.281001f)));
    rays.push_back(Ray(Point(-0.036607f, 0.144761f, 0.527258f), Vector(-0.953675f, -0.024133f, 0.299868f)));
    rays.push_back(Ray(Point(-0.295319f, 0.254636f, -1.0399f), Vector(0.830065f, -0.146705f, 0.538024f)));
    rays.push_back(Ray(Point(-1.013208f, 0.65831f, 0.468006f), Vector(0.267413f, -0.202862f, -0.941986f)));
    rays.push_back(Ray(Point(-0.356008f, 1.9899f, 0.613468f), Vector(0.728009f, -0.516699f, -0.450583f)));
    rays.push_back(Ray(Point(0.331174f, 0.6001f, 0.129577f), Vector(0.088287f, 0.788631f, -0.608495f)));
    rays.push_back(Ray(Point(0.178353f, 0.322839f, -1.0399f), Vector(0.189114f, -0.022466f, 0.981698f)));
    rays.push_back(Ray(Point(0.9999f, 1.857758f, -0.138231f), Vector(-0.709645f, 0.099015f, 0.697568f)));
    rays.push_back(Ray(Point(0.563589f, 1.9899f, -0.412589f), Vector(0.444875f, -0.723146f, 0.528343f)));
    rays.push_back(Ray(Point(0.333328f, 1.9899f, 0.528871f), Vector(0.812593f, -0.466611f, -0.349237f)));
    rays.push_back(Ray(Point(-1.011115f, 0.241725f, 0.933914f), Vector(0.805262f, 0.011356f, -0.59281f)));
    rays.push_back(Ray(Point(0.290514f, 1.9899f, 0.735056f), Vector(-0.245842f, -0.109166f, -0.963143f)));
    rays.push_back(Ray(Point(0.009018f, 0.962716f, -0.080111f), Vector(0.713414f, 0.36877f, 0.595859f)));
    rays.push_back(Ray(Point(0.021674f, 0.96401f, -0.084108f), Vector(-0.378158f, 0.663146f, 0.645937f)));
    rays.push_back(Ray(Point(0.029739f, 0.964276f, -0.086655f), Vector(0.472106f, -0.673637f, 0.56862f)));
    rays.push_back(Ray(Point(0.001507f, 0.962552f, -0.077739f), Vector(0.995733f, 0.070956f, 0.058997f)));
    rays.push_back(Ray(Point(0.005545f, 0.960906f, -0.079014f), Vector(0.885136f, -0.334335f, 0.323657f)));
    rays.push_back(Ray(Point(0.024848f, 0.972241f, -0.08511f), Vector(0.532642f, 0.632827f, 0.56198f)));
    rays.push_back(Ray(Point(0.025216f, 0.97656f, -0.085227f), Vector(0.249107f, 0.936061f, 0.248468f)));
    rays.push_back(Ray(Point(0.025053f, 0.98037f, -0.085175f), Vector(-0.8649f, -0.112189f, 0.489246f)));
    rays.push_back(Ray(Point(0.000777f, 0.967253f, -0.077509f), Vector(-0.038906f, 0.977317f, 0.20818f)));
    rays.push_back(Ray(Point(0.000976f, 0.975871f, -0.077571f), Vector(0.768465f, -0.527592f, 0.362088f)));
    rays.push_back(Ray(Point(0.000145f, 0.985003f, -0.077309f), Vector(0.85652f, -0.502436f, 0.118036f)));
    rays.push_back(Ray(Point(0.004852f, 0.965006f, -0.078795f), Vector(0.877387f, -0.478769f, -0.031192f)));
    rays.push_back(Ray(Point(0.004688f, 0.969382f, -0.078744f), Vector(0.821689f, 0.568697f, 0.037551f)));
    rays.push_back(Ray(Point(0.005036f, 0.974013f, -0.078853f), Vector(-0.191314f, 0.828856f, 0.525735f)));
    rays.push_back(Ray(Point(0.004189f, 0.982504f, -0.078586f), Vector(0.636681f, 0.666124f, 0.388481f)));
    rays.push_back(Ray(Point(0.004538f, 0.987395f, -0.078696f), Vector(-0.699341f, -0.639752f, 0.31881f)));
    rays.push_back(Ray(Point(0.024674f, 0.983434f, -0.085055f), Vector(0.843581f, 0.530085f, -0.085918f)));
    rays.push_back(Ray(Point(0.024322f, 0.98698f, -0.084944f), Vector(0.474572f, -0.627035f, 0.617745f)));
    rays.push_back(Ray(Point(0.024474f, 0.992305f, -0.084992f), Vector(0.678383f, -0.587703f, 0.44091f)));
    rays.push_back(Ray(Point(0.024117f, 0.997653f, -0.084879f), Vector(0.718106f, -0.570286f, 0.398869f)));
    rays.push_back(Ray(Point(0.029076f, 0.966057f, -0.086445f), Vector(0.809297f, -0.475396f, 0.345016f)));
    rays.push_back(Ray(Point(0.029425f, 0.970957f, -0.086556f), Vector(0.481305f, -0.830724f, 0.27972f)));
    rays.push_back(Ray(Point(0.028899f, 0.98512f, -0.086389f), Vector(0.401783f, 0.675812f, 0.617939f)));
    rays.push_back(Ray(Point(0.028556f, 0.986451f, -0.086281f), Vector(-0.8681f, 0.369521f, 0.331445f)));
    rays.push_back(Ray(Point(0.028177f, 0.99135f, -0.086162f), Vector(0.89603f, 0.36039f, -0.259325f)));
    rays.push_back(Ray(Point(0.02834f, 0.994195f, -0.086213f), Vector(0.772829f, 0.254965f, 0.581144f)));
    rays.push_back(Ray(Point(0.009066f, 0.975885f, -0.080126f), Vector(0.676975f, 0.635453f, 0.371355f)));
    rays.push_back(Ray(Point(0.008569f, 0.98908f, -0.079969f), Vector(-0.364155f, 0.689313f, 0.62629f)));
    rays.push_back(Ray(Point(0.008131f, 0.990594f, -0.079831f), Vector(0.42643f, 0.90366f, -0.039452f)));
    rays.push_back(Ray(Point(0.007782f, 0.994135f, -0.079721f), Vector(-0.435706f, 0.88085f, 0.185104f)));
    rays.push_back(Ray(Point(0.013398f, 0.965654f, -0.081494f), Vector(0.300823f, 0.953071f, -0.034071f)));
    rays.push_back(Ray(Point(0.013233f, 0.969904f, -0.081442f), Vector(0.906459f, -0.282852f, 0.313573f)));
    rays.push_back(Ray(Point(0.011974f, 0.997698f, -0.081044f), Vector(0.298163f, 0.778364f, 0.552493f)));
    rays.push_back(Ray(Point(0.017435f, 0.967656f, -0.082769f), Vector(0.718751f, 0.181466f, 0.671168f)));
    rays.push_back(Ray(Point(0.016768f, 0.972614f, -0.082558f), Vector(0.047405f, -0.844418f, 0.533583f)));
    rays.push_back(Ray(Point(0.016806f, 0.98441f, -0.082571f), Vector(0.350129f, -0.908819f, 0.226843f)));
    rays.push_back(Ray(Point(0.016396f, 0.986763f, -0.082441f), Vector(-0.170665f, 0.713841f, 0.679193f)));
    rays.push_back(Ray(Point(0.016043f, 0.99095f, -0.08233f), Vector(0.982759f, -0.026326f, -0.183009f)));
    rays.push_back(Ray(Point(0.020823f, 0.97043f, -0.083839f), Vector(0.323768f, 0.702959f, 0.633264f)));
    rays.push_back(Ray(Point(0.02045f, 0.985034f, -0.083721f), Vector(0.416546f, -0.797393f, 0.43664f)));
    rays.push_back(Ray(Point(0.020616f, 0.989544f, -0.083773f), Vector(0.854979f, -0.392205f, 0.339392f)));
    rays.push_back(Ray(Point(0.020248f, 0.993283f, -0.083657f), Vector(0.987663f, 0.016197f, -0.155757f)));
    rays.push_back(Ray(Point(0.019929f, 0.994867f, -0.083556f), Vector(0.9874f, 0.152739f, 0.041368f)));
    rays.push_back(Ray(Point(-0.08723f, 1.9899f, 0.575471f), Vector(0.855014f, -0.513268f, 0.074209f)));
    rays.push_back(Ray(Point(0.9999f, 0.202638f, -0.128422f), Vector(-0.706867f, 0.618145f, -0.343855f)));
    rays.push_back(Ray(Point(0.9999f, 1.242888f, -0.367192f), Vector(-0.530554f, 0.842208f, -0.095904f)));
    rays.push_back(Ray(Point(-0.060222f, 1.9799f, 0.062737f), Vector(-0.926859f, -0.071713f, -0.368497f)));
    rays.push_back(Ray(Point(0.9999f, 1.297075f, 0.362705f), Vector(-0.881669f, 0.452134f, -0.135034f)));
    rays.push_back(Ray(Point(0.923828f, 1.9899f, -0.087266f), Vector(0.472341f, -0.138108f, 0.870528f)));
    rays.push_back(Ray(Point(0.474502f, 0.6001f, 0.154897f), Vector(0.723463f, 0.09135f, -0.684293f)));
    rays.push_back(Ray(Point(0.554944f, 1.9899f, -0.240485f), Vector(0.41214f, -0.675607f, 0.611306f)));
    rays.push_back(Ray(Point(0.399675f, 0.6001f, 0.613953f), Vector(0.600014f, 0.742674f, 0.297352f)));
    rays.push_back(Ray(Point(0.9999f, 1.497201f, 0.600343f), Vector(-0.861817f, -0.47105f, -0.188104f)));
    rays.push_back(Ray(Point(-0.301804f, 1.9899f, 0.337268f), Vector(0.913796f, -0.406037f, -0.010523f)));
    rays.push_back(Ray(Point(0.358598f, 0.6001f, 0.37706f), Vector(-0.054451f, 0.29928f, -0.95261f)));
    rays.push_back(Ray(Point(-0.627033f, 0.0001f, 0.649178f), Vector(0.320664f, 0.641156f, -0.697204f)));
    rays.push_back(Ray(Point(-0.597958f, 1.9899f, 0.521849f), Vector(0.273549f, -0.474044f, -0.836931f)));
    rays.push_back(Ray(Point(0.9999f, 1.623055f, -0.069328f), Vector(-0.728318f, -0.535092f, -0.428053f)));
    rays.push_back(Ray(Point(0.9999f, 1.23877f, 0.615227f), Vector(-0.879412f, -0.282403f, -0.383253f)));
    rays.push_back(Ray(Point(-1.013706f, 0.757368f, 0.424118f), Vector(0.449241f, -0.25603f, -0.855939f)));
    rays.push_back(Ray(Point(0.9999f, 0.939995f, -0.362141f), Vector(-0.961691f, -0.267098f, -0.06172f)));
    rays.push_back(Ray(Point(-1.012805f, 0.578053f, 0.871119f), Vector(0.049408f, 0.983123f, -0.176147f)));
    rays.push_back(Ray(Point(-0.315754f, 1.9899f, 0.066318f), Vector(-0.812783f, -0.273546f, -0.51435f)));
    rays.push_back(Ray(Point(0.321691f, 0.6001f, 0.158978f), Vector(-0.771613f, 0.417085f, -0.480263f)));
    rays.push_back(Ray(Point(0.9999f, 1.799586f, 0.469271f), Vector(-0.489647f, -0.344573f, -0.800947f)));
    rays.push_back(Ray(Point(0.9999f, 1.730928f, 0.177543f), Vector(-0.601296f, -0.639367f, -0.479221f)));
    rays.push_back(Ray(Point(-1.015188f, 1.05238f, 0.454325f), Vector(0.977032f, -0.195402f, 0.08501f)));
    rays.push_back(Ray(Point(0.048103f, 0.6001f, 0.356522f), Vector(0.265343f, 0.337964f, -0.902981f)));
    rays.push_back(Ray(Point(0.405195f, 0.6001f, 0.348554f), Vector(0.719141f, 0.59688f, -0.355766f)));
    rays.push_back(Ray(Point(0.566487f, 0.6001f, 0.288491f), Vector(0.835603f, 0.374043f, 0.402318f)));
    rays.push_back(Ray(Point(-1.01789f, 1.589977f, 0.536692f), Vector(0.775561f, -0.26242f, -0.574143f)));
    rays.push_back(Ray(Point(-1.016505f, 1.314381f, 0.360997f), Vector(0.865207f, -0.163292f, -0.47408f)));
    rays.push_back(Ray(Point(-1.014594f, 0.934159f, 0.960433f), Vector(0.182773f, -0.356967f, -0.916061f)));
    rays.push_back(Ray(Point(-0.992537f, 0.364896f, -1.0399f), Vector(-0.719095f, 0.469416f, 0.512397f)));
    rays.push_back(Ray(Point(0.698607f, 0.0001f, 0.246804f), Vector(-0.380132f, 0.348096f, 0.85693f)));
    rays.push_back(Ray(Point(-1.010168f, 0.120124f, 0.833411f), Vector(0.513137f, 0.84375f, -0.157406f)));
    rays.push_back(Ray(Point(0.9999f, 0.025702f, -0.200312f), Vector(-0.842189f, 0.431779f, 0.322933f)));
    rays.push_back(Ray(Point(-0.585112f, 0.333553f, -1.0399f), Vector(0.282406f, 0.829969f, 0.481038f)));
    rays.push_back(Ray(Point(-0.252336f, 1.9899f, 0.577714f), Vector(0.935653f, -0.344871f, 0.074952f)));
    rays.push_back(Ray(Point(0.549229f, 1.9899f, -0.162089f), Vector(-0.989254f, -0.126064f, 0.074061f)));
    rays.push_back(Ray(Point(0.9999f, 1.869018f, -0.864785f), Vector(-0.220785f, 0.065077f, -0.973149f)));
    rays.push_back(Ray(Point(0.9999f, 0.854463f, 0.359756f), Vector(-0.515281f, 0.368487f, -0.773759f)));
    rays.push_back(Ray(Point(-1.019661f, 1.942384f, -0.126035f), Vector(0.866457f, 0.284332f, -0.410375f)));
    rays.push_back(Ray(Point(-0.750382f, 1.884639f, -1.0399f), Vector(-0.665927f, -0.686612f, 0.291728f)));
    rays.push_back(Ray(Point(0.90487f, 0.0001f, 0.719558f), Vector(-0.104949f, 0.60824f, -0.786784f)));
    rays.push_back(Ray(Point(0.393866f, 0.29724f, 0.078593f), Vector(0.92897f, 0.177859f, -0.324624f)));
    rays.push_back(Ray(Point(0.9999f, 1.607081f, 0.242348f), Vector(-0.494566f, 0.667765f, -0.556322f)));
    rays.push_back(Ray(Point(0.091287f, 0.257273f, 0.122258f), Vector(-0.242794f, -0.738544f, -0.628971f)));
    rays.push_back(Ray(Point(-0.162568f, 1.788709f, -1.0399f), Vector(-0.973103f, 0.015739f, 0.229834f)));
    rays.push_back(Ray(Point(0.026701f, 1.033211f, -0.085695f), Vector(0.408175f, -0.624942f, 0.665463f)));
    rays.push_back(Ray(Point(0.030928f, 1.032393f, -0.08703f), Vector(0.123324f, 0.759912f, 0.63822f)));
    rays.push_back(Ray(Point(-0.001715f, 1.034101f, -0.076722f), Vector(-0.667503f, 0.588478f, 0.456217f)));
    rays.push_back(Ray(Point(-0.001623f, 1.038538f, -0.076751f), Vector(0.024064f, -0.902436f, 0.430152f)));
    rays.push_back(Ray(Point(0.002552f, 1.035296f, -0.078069f), Vector(-0.11647f, 0.693287f, 0.711187f)));
    rays.push_back(Ray(Point(0.014532f, 1.034632f, -0.081852f), Vector(-0.294593f, 0.884132f, 0.362666f)));
    rays.push_back(Ray(Point(0.018412f, 1.035489f, -0.083077f), Vector(0.947444f, 0.18172f, -0.263303f)));
    rays.push_back(Ray(Point(0.022497f, 1.03841f, -0.084368f), Vector(0.959964f, 0.007516f, 0.280022f)));
    rays.push_back(Ray(Point(0.006605f, 1.032405f, -0.079349f), Vector(0.985003f, -0.102191f, -0.139019f)));
    rays.push_back(Ray(Point(0.010374f, 1.033661f, -0.080539f), Vector(0.274354f, 0.769726f, 0.576413f)));
    rays.push_back(Ray(Point(0.010473f, 1.03791f, -0.080571f), Vector(0.985599f, -0.158249f, -0.059595f)));
    rays.push_back(Ray(Point(-0.803562f, 0.0001f, -0.228902f), Vector(-0.178168f, 0.389503f, 0.903628f)));
    rays.push_back(Ray(Point(0.34599f, 1.606725f, -1.0399f), Vector(0.809635f, -0.586269f, 0.027933f)));
    rays.push_back(Ray(Point(0.9999f, 0.85329f, 0.031739f), Vector(-0.029876f, -0.27042f, -0.962279f)));
    rays.push_back(Ray(Point(-0.22574f, 0.0001f, 0.678843f), Vector(-0.708702f, 0.686713f, -0.161762f)));
    rays.push_back(Ray(Point(0.023613f, 1.008032f, -0.08472f), Vector(0.959711f, 0.156442f, 0.23341f)));
    rays.push_back(Ray(Point(0.023279f, 1.012906f, -0.084615f), Vector(-0.791907f, -0.049025f, 0.608671f)));
    rays.push_back(Ray(Point(0.023368f, 1.016967f, -0.084643f), Vector(0.248051f, 0.968655f, 0.01336f)));
    rays.push_back(Ray(Point(0.9999f, 1.920189f, 0.708497f), Vector(-0.75467f, 0.478602f, -0.448791f)));
    rays.push_back(Ray(Point(0.023187f, 1.024839f, -0.084585f), Vector(-0.620776f, 0.758178f, 0.199508f)));
    rays.push_back(Ray(Point(0.022905f, 1.029155f, -0.084497f), Vector(0.755908f, 0.649641f, -0.081055f)));
    rays.push_back(Ray(Point(0.02799f, 0.998062f, -0.086102f), Vector(0.867166f, -0.266193f, 0.420909f)));
    rays.push_back(Ray(Point(0.027672f, 1.006053f, -0.086002f), Vector(0.392482f, -0.550871f, 0.736545f)));
    rays.push_back(Ray(Point(0.015509f, 1.006059f, -0.082161f), Vector(0.749089f, -0.661434f, -0.03701f)));
    rays.push_back(Ray(Point(0.015664f, 1.009729f, -0.08221f), Vector(0.992101f, 0.109306f, 0.061546f)));
    rays.push_back(Ray(Point(0.014969f, 1.027317f, -0.08199f), Vector(0.540315f, -0.768436f, 0.342879f)));
    rays.push_back(Ray(Point(0.020087f, 0.999764f, -0.083606f), Vector(-0.707806f, 0.324578f, 0.627424f)));
    rays.push_back(Ray(Point(0.019206f, 1.017806f, -0.083329f), Vector(0.928285f, 0.355998f, -0.107484f)));
    rays.push_back(Ray(Point(0.019293f, 1.019329f, -0.083356f), Vector(-0.720309f, -0.37911f, 0.580888f)));
    rays.push_back(Ray(Point(-0.000469f, 0.998608f, -0.077115f), Vector(0.726136f, 0.675683f, 0.1272f)));
    rays.push_back(Ray(Point(-0.000313f, 1.003047f, -0.077165f), Vector(0.959575f, -0.046157f, -0.277643f)));
    rays.push_back(Ray(Point(-0.000656f, 1.007224f, -0.077056f), Vector(0.834414f, 0.47718f, 0.275776f)));
    rays.push_back(Ray(Point(-0.0005f, 1.011662f, -0.077106f), Vector(0.336975f, -0.560988f, 0.756135f)));
    rays.push_back(Ray(Point(-0.000843f, 1.016355f, -0.076997f), Vector(-0.856136f, -0.292403f, 0.426065f)));
    rays.push_back(Ray(Point(-0.001249f, 1.020788f, -0.076869f), Vector(0.825439f, 0.545125f, -0.14659f)));
    rays.push_back(Ray(Point(-0.001029f, 1.024968f, -0.076938f), Vector(0.581591f, 0.809851f, -0.076765f)));
    rays.push_back(Ray(Point(0.003721f, 1.00004f, -0.078439f), Vector(0.549083f, -0.584247f, 0.597631f)));
    rays.push_back(Ray(Point(0.003513f, 1.010074f, -0.078373f), Vector(0.875706f, -0.3303f, 0.352193f)));
    rays.push_back(Ray(Point(0.003155f, 1.013418f, -0.078259f), Vector(-0.465993f, 0.87142f, 0.153225f)));
    rays.push_back(Ray(Point(0.003314f, 1.017535f, -0.07831f), Vector(0.80288f, 0.565503f, -0.188653f)));
    rays.push_back(Ray(Point(0.002656f, 1.022935f, -0.078102f), Vector(0.979688f, -0.161992f, -0.118196f)));
    rays.push_back(Ray(Point(0.00275f, 1.026795f, -0.078132f), Vector(0.962606f, 0.262889f, 0.065412f)));
    rays.push_back(Ray(Point(0.02778f, 1.006287f, -0.086036f), Vector(0.35193f, 0.87553f, 0.331047f)));
    rays.push_back(Ray(Point(0.027485f, 1.010671f, -0.085943f), Vector(0.869092f, 0.176048f, 0.462262f)));
    rays.push_back(Ray(Point(0.027581f, 1.016095f, -0.085973f), Vector(0.40036f, -0.60126f, 0.691519f)));
    rays.push_back(Ray(Point(0.285035f, 0.6001f, 0.220832f), Vector(-0.761967f, 0.472848f, -0.442517f)));
    rays.push_back(Ray(Point(0.9999f, 1.033705f, -0.018579f), Vector(-0.837537f, -0.545735f, -0.026559f)));
    rays.push_back(Ray(Point(0.9999f, 0.585279f, 0.284616f), Vector(-0.955041f, 0.141879f, -0.260321f)));
    rays.push_back(Ray(Point(-0.039539f, 1.9799f, 0.138218f), Vector(-0.514755f, -0.756723f, -0.402986f)));
    rays.push_back(Ray(Point(0.548451f, 0.6001f, 0.18039f), Vector(0.858298f, 0.475219f, -0.193626f)));
    rays.push_back(Ray(Point(0.9999f, 0.398487f, 0.06048f), Vector(-0.234113f, 0.96851f, -0.084732f)));
    rays.push_back(Ray(Point(0.9999f, 1.658244f, -0.033258f), Vector(-0.75185f, -0.274058f, -0.599678f)));
    rays.push_back(Ray(Point(-0.229472f, 1.9899f, 0.565577f), Vector(-0.652909f, -0.61426f, -0.443163f)));
    rays.push_back(Ray(Point(0.9999f, 1.906041f, 0.463451f), Vector(-0.258725f, 0.112968f, -0.959322f)));
    rays.push_back(Ray(Point(-0.520205f, 1.9899f, 0.829438f), Vector(-0.299392f, -0.953276f, 0.040366f)));
    rays.push_back(Ray(Point(0.336718f, 1.9899f, -0.118113f), Vector(0.523481f, -0.601342f, -0.603618f)));
    rays.push_back(Ray(Point(-0.619186f, 0.0001f, 0.592889f), Vector(-0.324601f, 0.818703f, -0.473667f)));
    rays.push_back(Ray(Point(-0.623373f, 0.0001f, 0.492592f), Vector(0.17217f, 0.940783f, -0.292036f)));
    rays.push_back(Ray(Point(0.550528f, 0.0001f, -0.033355f), Vector(-0.546569f, 0.134257f, -0.826582f)));
    rays.push_back(Ray(Point(0.392089f, 1.9899f, 0.623304f), Vector(-0.409304f, -0.349247f, -0.84291f)));
    rays.push_back(Ray(Point(0.881483f, 1.9899f, 0.818708f), Vector(-0.762594f, -0.610163f, -0.21483f)));
    rays.push_back(Ray(Point(0.294915f, 1.9899f, 0.183781f), Vector(0.371037f, -0.91943f, -0.130307f)));
    rays.push_back(Ray(Point(0.9999f, 1.596304f, -0.184391f), Vector(-0.710702f, 0.593103f, 0.378328f)));
    rays.push_back(Ray(Point(0.317208f, 0.6001f, 0.296303f), Vector(-0.250535f, 0.593179f, -0.765095f)));
    rays.push_back(Ray(Point(0.47731f, 0.6001f, 0.209325f), Vector(-0.585608f, 0.570342f, -0.575997f)));
    rays.push_back(Ray(Point(0.524842f, 0.6001f, 0.193247f), Vector(0.311474f, 0.223455f, -0.923608f)));
    rays.push_back(Ray(Point(0.652238f, 0.6001f, 0.179219f), Vector(0.099578f, 0.983445f, -0.151395f)));
    rays.push_back(Ray(Point(0.24435f, 0.6001f, 0.038352f), Vector(0.880039f, 0.148154f, -0.4512f)));
    rays.push_back(Ray(Point(-0.974644f, 0.0001f, 0.599948f), Vector(-0.024956f, 0.390937f, -0.920079f)));
    rays.push_back(Ray(Point(0.626319f, 1.9899f, 0.832438f), Vector(-0.996454f, -0.051096f, 0.066843f)));
    rays.push_back(Ray(Point(-1.017094f, 1.431592f, 0.312991f), Vector(0.344924f, -0.854309f, -0.388822f)));
    rays.push_back(Ray(Point(0.9999f, 1.382225f, -0.367423f), Vector(-0.849155f, -0.307471f, 0.429414f)));
    rays.push_back(Ray(Point(0.9999f, 1.314756f, 0.644446f), Vector(-0.639728f, -0.726995f, -0.249454f)));
    rays.push_back(Ray(Point(0.027282f, 1.019449f, -0.085879f), Vector(0.319527f, 0.728985f, 0.605379f)));
    rays.push_back(Ray(Point(0.026975f, 1.02645f, -0.085782f), Vector(0.939786f, -0.301603f, 0.16074f)));
    rays.push_back(Ray(Point(0.027068f, 1.030189f, -0.085811f), Vector(-0.805713f, 0.508681f, 0.303431f)));
    rays.push_back(Ray(Point(0.007945f, 1, -0.079772f), Vector(0.848238f, -0.493199f, 0.192995f)));
    rays.push_back(Ray(Point(0.007561f, 1.00225f, -0.079651f), Vector(0.067226f, 0.992902f, 0.098111f)));
    rays.push_back(Ray(Point(0.007381f, 1.010873f, -0.079594f), Vector(0.36905f, -0.845918f, 0.385f)));
    rays.push_back(Ray(Point(0.007229f, 1.022581f, -0.079546f), Vector(-0.358034f, 0.695531f, 0.622935f)));
    rays.push_back(Ray(Point(0.011644f, 1.000657f, -0.080941f), Vector(0.976043f, -0.115032f, -0.184684f)));
    rays.push_back(Ray(Point(0.011809f, 1.004906f, -0.080992f), Vector(0.287869f, 0.848476f, 0.444095f)));
    rays.push_back(Ray(Point(0.011444f, 1.00864f, -0.080877f), Vector(0.924053f, 0.273152f, 0.267423f)));
    rays.push_back(Ray(Point(0.011595f, 1.012156f, -0.080925f), Vector(0.691155f, -0.606963f, 0.392302f)));
    rays.push_back(Ray(Point(0.010877f, 1.021423f, -0.080698f), Vector(-0.42534f, 0.659155f, 0.620162f)));
    rays.push_back(Ray(Point(0.011098f, 1.025157f, -0.080768f), Vector(-0.740609f, 0.105101f, 0.663666f)));
    rays.push_back(Ray(Point(0.01066f, 1.030694f, -0.08063f), Vector(0.843746f, 0.265677f, 0.466377f)));
    rays.push_back(Ray(Point(0.9999f, 1.215727f, 0.834748f), Vector(-0.230257f, 0.964624f, -0.128383f)));
    rays.push_back(Ray(Point(0.047259f, 0.427792f, 0.261682f), Vector(-0.972996f, 0.025839f, 0.229371f)));
    rays.push_back(Ray(Point(0.164903f, 0.6001f, 0.013379f), Vector(-0.377893f, 0.915443f, 0.138423f)));
    rays.push_back(Ray(Point(-0.223458f, 1.9899f, 0.872102f), Vector(0.657926f, -0.743906f, -0.117209f)));
    rays.push_back(Ray(Point(0.059482f, 0.0001f, -0.010915f), Vector(-0.333338f, 0.933918f, -0.129166f)));
    rays.push_back(Ray(Point(0.214182f, 0.0001f, -0.116444f), Vector(0.590594f, 0.805975f, -0.040032f)));
    rays.push_back(Ray(Point(0.221586f, 0.6001f, 0.127118f), Vector(-0.669188f, 0.127368f, -0.732096f)));
    rays.push_back(Ray(Point(0.9999f, 0.540271f, 0.305003f), Vector(-0.04972f, 0.658403f, -0.751021f)));
    rays.push_back(Ray(Point(0.9999f, 1.009351f, -0.238167f), Vector(-0.466995f, -0.508425f, -0.723478f)));
    rays.push_back(Ray(Point(0.9999f, 1.146473f, -0.042495f), Vector(-0.034817f, -0.614269f, -0.788328f)));
    rays.push_back(Ray(Point(0.486772f, 1.9899f, -0.942847f), Vector(0.764866f, -0.149666f, 0.626562f)));
    rays.push_back(Ray(Point(0.536214f, 0.0001f, -0.412448f), Vector(0.043821f, 0.998941f, -0.014026f)));
    rays.push_back(Ray(Point(-0.147943f, 0.868808f, -0.030545f), Vector(0.54467f, -0.458618f, 0.702142f)));
    rays.push_back(Ray(Point(-0.077905f, 0.331392f, -0.052662f), Vector(0.979314f, 0.141942f, -0.14421f)));
    rays.push_back(Ray(Point(-0.57006f, 1.2001f, -0.27711f), Vector(-0.142338f, 0.952998f, -0.267458f)));
    rays.push_back(Ray(Point(0.9999f, 0.850112f, 0.078524f), Vector(-0.377811f, 0.798283f, 0.469045f)));
    rays.push_back(Ray(Point(0.615192f, 1.9899f, -0.078757f), Vector(0.905769f, -0.410253f, 0.106184f)));
    rays.push_back(Ray(Point(0.525337f, 0.0001f, -0.279773f), Vector(-0.302174f, 0.306659f, 0.90258f)));
    rays.push_back(Ray(Point(-0.262308f, 1.198156f, -1.0399f), Vector(-0.227344f, 0.354618f, 0.906951f)));
    rays.push_back(Ray(Point(-0.03847f, 0.794797f, -0.065115f), Vector(0.954147f, -0.273367f, -0.12196f)));
    rays.push_back(Ray(Point(0.756455f, 0.0001f, 0.178396f), Vector(0.950173f, 0.29851f, -0.089788f)));
    rays.push_back(Ray(Point(-1.010143f, 0.476303f, 0.285791f), Vector(0.030246f, 0.940203f, -0.339268f)));
    rays.push_back(Ray(Point(0.9999f, 0.2427f, -0.063842f), Vector(-0.945981f, 0.090541f, -0.311323f)));
    rays.push_back(Ray(Point(-0.120372f, 1.636137f, -1.0399f), Vector(-0.988087f, 0.066585f, 0.138747f)));
    rays.push_back(Ray(Point(0.940738f, 0.898469f, -1.0399f), Vector(0.370335f, 0.928659f, 0.021096f)));
    rays.push_back(Ray(Point(0.792971f, 1.9899f, -0.034748f), Vector(-0.505268f, -0.862894f, 0.010906f)));
    rays.push_back(Ray(Point(0.9999f, 0.727313f, -0.349072f), Vector(-0.508153f, 0.80277f, -0.311995f)));
    rays.push_back(Ray(Point(-0.605068f, 0.41085f, -0.15155f), Vector(-0.738017f, -0.111108f, -0.665572f)));
    rays.push_back(Ray(Point(0.370941f, 0.6001f, 0.399191f), Vector(-0.33745f, 0.940816f, 0.031517f)));
    rays.push_back(Ray(Point(0.807612f, 1.9899f, -0.249529f), Vector(0.653874f, -0.034849f, 0.7558f)));
    rays.push_back(Ray(Point(0.056459f, 0.0001f, -0.170304f), Vector(-0.807819f, 0.174749f, 0.562931f)));
    rays.push_back(Ray(Point(-0.288182f, 0.0001f, 0.22932f), Vector(0.94005f, 0.154916f, 0.303821f)));
    rays.push_back(Ray(Point(0.9999f, 1.227964f, -0.882933f), Vector(-0.276689f, 0.054856f, -0.959393f)));
    rays.push_back(Ray(Point(0.913771f, 0.0001f, 0.087109f), Vector(0.897225f, 0.059806f, -0.437505f)));
    rays.push_back(Ray(Point(-1.014917f, 0.998458f, 0.015282f), Vector(0.100314f, 0.622798f, -0.775925f)));
    rays.push_back(Ray(Point(-0.415586f, 1.300735f, -1.0399f), Vector(-0.141571f, 0.934343f, 0.327048f)));
    rays.push_back(Ray(Point(-0.874042f, 1.841374f, -1.0399f), Vector(-0.28312f, -0.884901f, 0.369856f)));
    rays.push_back(Ray(Point(0.529252f, 1.9899f, -0.452267f), Vector(-0.699122f, -0.588141f, -0.406595f)));
    rays.push_back(Ray(Point(-1.019476f, 1.905673f, -0.31868f), Vector(0.883947f, -0.466271f, -0.035043f)));
    rays.push_back(Ray(Point(-0.341688f, 0.807941f, 0.030638f), Vector(0.657522f, -0.752854f, -0.029592f)));
    rays.push_back(Ray(Point(0.610653f, 0.0001f, 0.878091f), Vector(-0.81289f, 0.524934f, -0.252298f)));
    rays.push_back(Ray(Point(-0.941846f, 1.9899f, 0.618139f), Vector(0.261806f, -0.649939f, -0.713468f)));
    rays.push_back(Ray(Point(-1.01871f, 1.753284f, -0.378595f), Vector(0.443664f, -0.148055f, -0.883879f)));
    rays.push_back(Ray(Point(0.07723f, 1.150287f, -1.0399f), Vector(-0.842533f, 0.347278f, 0.411748f)));
    rays.push_back(Ray(Point(0.9999f, 0.64935f, 0.629664f), Vector(-0.65263f, -0.741053f, 0.157845f)));
    rays.push_back(Ray(Point(0.9999f, 1.967628f, 0.05312f), Vector(-0.1082f, 0.993192f, -0.043162f)));
    rays.push_back(Ray(Point(0.9999f, 1.260335f, 0.419643f), Vector(-0.549168f, -0.811671f, -0.199008f)));
    rays.push_back(Ray(Point(0.9999f, 1.093782f, 0.054299f), Vector(-0.012999f, 0.551528f, -0.834055f)));
    rays.push_back(Ray(Point(-1.007771f, 0.399798f, 0.16212f), Vector(0.428568f, 0.755833f, -0.495021f)));
    rays.push_back(Ray(Point(0.9999f, 1.411455f, 0.322277f), Vector(-0.37561f, -0.057342f, -0.925002f)));
    rays.push_back(Ray(Point(0.277599f, 1.045296f, -1.0399f), Vector(0.796186f, -0.310157f, 0.51951f)));
    rays.push_back(Ray(Point(0.997354f, 0.0001f, -0.324151f), Vector(0.608366f, 0.672045f, -0.422193f)));
    rays.push_back(Ray(Point(-0.08747f, 1.105258f, -1.0399f), Vector(0.116995f, 0.962266f, 0.245674f)));
    rays.push_back(Ray(Point(-0.651665f, 0.409658f, -1.0399f), Vector(0.022999f, -0.405509f, 0.913802f)));
    rays.push_back(Ray(Point(0.951163f, 0.0001f, 0.560381f), Vector(-0.359402f, 0.140418f, -0.922558f)));
    rays.push_back(Ray(Point(-0.410453f, 1.2001f, 0.028917f), Vector(0.938201f, 0.156729f, -0.308568f)));
    rays.push_back(Ray(Point(-0.615467f, 0.15464f, -1.0399f), Vector(-0.709916f, 0.687299f, 0.153754f)));
    rays.push_back(Ray(Point(0.309648f, 0.6001f, 0.375605f), Vector(0.941575f, 0.237865f, -0.238446f)));
    rays.push_back(Ray(Point(-1.019606f, 1.931575f, 0.619043f), Vector(0.590776f, 0.731526f, 0.340374f)));
    rays.push_back(Ray(Point(-0.157835f, 1.9899f, 0.901284f), Vector(0.144324f, -0.699505f, -0.699902f)));
    rays.push_back(Ray(Point(0.81508f, 1.9899f, 0.731699f), Vector(-0.32696f, -0.093107f, -0.94044f)));
    rays.push_back(Ray(Point(-1.012058f, 0.455927f, 0.511426f), Vector(0.924457f, 0.293263f, -0.243672f)));
    rays.push_back(Ray(Point(0.9999f, 0.606556f, 0.654143f), Vector(-0.743685f, -0.151408f, -0.651159f)));
    rays.push_back(Ray(Point(0.503156f, 0.0001f, -0.068613f), Vector(0.459973f, 0.216503f, 0.861133f)));
    rays.push_back(Ray(Point(-0.061687f, 0.339666f, -0.057783f), Vector(-0.70111f, 0.391308f, 0.59609f)));
    rays.push_back(Ray(Point(0.9999f, 1.072494f, -0.169708f), Vector(-0.91677f, -0.35061f, 0.191323f)));
    rays.push_back(Ray(Point(0.026935f, 0.637154f, -0.085769f), Vector(0.117233f, 0.901377f, 0.416865f)));
    rays.push_back(Ray(Point(0.910856f, 1.719405f, -1.0399f), Vector(-0.489281f, -0.833544f, 0.256533f)));
    rays.push_back(Ray(Point(0.482328f, 0.445861f, -1.0399f), Vector(0.49897f, 0.866311f, 0.023099f)));
    rays.push_back(Ray(Point(0.955844f, 0.369212f, -1.0399f), Vector(-0.79676f, -0.035786f, 0.603236f)));
    rays.push_back(Ray(Point(0.9999f, 1.223758f, -0.355869f), Vector(-0.258638f, 0.425849f, -0.86704f)));
    rays.push_back(Ray(Point(0.9999f, 1.046063f, 0.20077f), Vector(-0.039669f, -0.172279f, -0.984249f)));
    rays.push_back(Ray(Point(0.9999f, 0.929343f, -0.219553f), Vector(-0.026471f, 0.209348f, -0.977483f)));
    rays.push_back(Ray(Point(-0.265213f, 0.0001f, 0.209002f), Vector(0.820247f, 0.382036f, -0.425728f)));
    rays.push_back(Ray(Point(0.351242f, 1.9899f, 0.63562f), Vector(-0.714474f, -0.328284f, -0.617864f)));
    rays.push_back(Ray(Point(0.9999f, 1.929564f, 0.098141f), Vector(-0.041481f, -0.245271f, -0.968567f)));
    rays.push_back(Ray(Point(0.9999f, 0.95493f, -0.366595f), Vector(-0.167568f, 0.955171f, 0.24407f)));
    rays.push_back(Ray(Point(0.9999f, 1.579473f, 0.253663f), Vector(-0.171772f, 0.936498f, 0.305724f)));
    rays.push_back(Ray(Point(0.246778f, 0.6001f, 0.477758f), Vector(-0.949025f, 0.044961f, 0.311977f)));
    rays.push_back(Ray(Point(-1.013269f, 0.670539f, 0.426898f), Vector(0.942071f, -0.299352f, -0.151296f)));
    rays.push_back(Ray(Point(-0.094618f, 0.0001f, 0.956391f), Vector(0.131917f, 0.977049f, -0.167252f)));
    rays.push_back(Ray(Point(0.9999f, 0.634218f, 0.322396f), Vector(-0.872944f, -0.487792f, -0.005293f)));
    rays.push_back(Ray(Point(0.9999f, 1.719544f, -0.312502f), Vector(-0.105931f, 0.846248f, -0.522152f)));
    rays.push_back(Ray(Point(-0.383326f, 0.0001f, 0.152156f), Vector(0.154913f, 0.940589f, -0.302151f)));
    rays.push_back(Ray(Point(0.9999f, 0.858024f, -0.198428f), Vector(-0.679496f, 0.584505f, 0.443439f)));
    rays.push_back(Ray(Point(0.284283f, 0.0001f, -0.138098f), Vector(-0.76774f, 0.3078f, 0.561991f)));
    rays.push_back(Ray(Point(0.9999f, 0.136768f, -0.130801f), Vector(-0.213162f, 0.238937f, -0.947349f)));
    rays.push_back(Ray(Point(0.9999f, 1.118179f, -0.021146f), Vector(-0.521889f, -0.509542f, 0.684104f)));
    rays.push_back(Ray(Point(0.171303f, 0.6001f, 0.381f), Vector(-0.507264f, 0.801719f, 0.316119f)));
    rays.push_back(Ray(Point(0.9999f, 0.487624f, -0.21139f), Vector(-0.665339f, 0.189369f, -0.722124f)));
    rays.push_back(Ray(Point(-0.688038f, 1.9899f, -0.498795f), Vector(-0.925059f, -0.208195f, -0.31768f)));
    rays.push_back(Ray(Point(0.460414f, 1.9899f, 0.748285f), Vector(-0.452175f, -0.683141f, -0.57346f)));
    rays.push_back(Ray(Point(0.9999f, 1.815607f, -0.033645f), Vector(-0.087199f, 0.79831f, -0.595901f)));
    rays.push_back(Ray(Point(-0.769957f, 1.9899f, 0.985279f), Vector(-0.523482f, -0.340652f, -0.780976f)));
    rays.push_back(Ray(Point(-0.865499f, 0.0001f, 0.901828f), Vector(-0.390934f, 0.920022f, 0.027017f)));
    rays.push_back(Ray(Point(-0.108958f, 0.0001f, 0.32411f), Vector(-0.743184f, 0.577566f, -0.33778f)));
    rays.push_back(Ray(Point(-0.454703f, 0.0001f, 0.34113f), Vector(-0.457361f, 0.862936f, -0.214854f)));
    rays.push_back(Ray(Point(-0.698518f, 0.0001f, 0.522222f), Vector(-0.3909f, 0.831122f, 0.395515f)));
    rays.push_back(Ray(Point(0.9999f, 1.393942f, -0.196892f), Vector(-0.003004f, 0.871673f, -0.490079f)));
    rays.push_back(Ray(Point(-1.012294f, 0.476336f, 0.74864f), Vector(0.897587f, 0.40833f, -0.166145f)));
    rays.push_back(Ray(Point(-0.005074f, 0.0001f, 0.406795f), Vector(0.890735f, 0.207604f, -0.40434f)));
    rays.push_back(Ray(Point(0.9999f, 0.42318f, 0.145944f), Vector(-0.59734f, -0.647429f, 0.473309f)));
    rays.push_back(Ray(Point(0.27254f, 1.9899f, -0.071222f), Vector(0.934626f, -0.310066f, -0.174165f)));
    rays.push_back(Ray(Point(-0.767061f, 1.9899f, 0.169389f), Vector(-0.538328f, -0.841936f, 0.036684f)));
    rays.push_back(Ray(Point(0.9999f, 0.699685f, 0.385696f), Vector(-0.706733f, 0.64116f, 0.299069f)));
    rays.push_back(Ray(Point(0.316975f, 0.6001f, 0.456913f), Vector(0.772327f, 0.524795f, 0.357912f)));
    rays.push_back(Ray(Point(0.9999f, 1.207669f, 0.431329f), Vector(-0.803559f, -0.595216f, 0.003461f)));
    rays.push_back(Ray(Point(0.304645f, 0.6001f, 0.392584f), Vector(0.766337f, 0.5558f, 0.322203f)));
    rays.push_back(Ray(Point(0.452691f, 1.9899f, 0.720107f), Vector(0.391943f, -0.849505f, -0.35316f)));
    rays.push_back(Ray(Point(-1.018394f, 1.690298f, 0.307947f), Vector(0.285927f, 0.000413f, -0.958251f)));
    rays.push_back(Ray(Point(0.073762f, 1.9799f, 0.016963f), Vector(-0.781217f, -0.400771f, -0.478626f)));
    rays.push_back(Ray(Point(-0.213414f, 0.0001f, 0.48477f), Vector(-0.432009f, 0.228928f, -0.872331f)));
    rays.push_back(Ray(Point(0.9999f, 0.884174f, -0.267955f), Vector(-0.942243f, -0.175998f, 0.284962f)));
    rays.push_back(Ray(Point(0.346029f, 1.9899f, 0.434608f), Vector(-0.883863f, -0.465803f, -0.042593f)));
    rays.push_back(Ray(Point(0.480921f, 0.6001f, 0.185466f), Vector(-0.233926f, 0.70421f, -0.670349f)));
    rays.push_back(Ray(Point(-0.961448f, 1.9899f, -0.260419f), Vector(0.066066f, -0.368455f, -0.927295f)));
    rays.push_back(Ray(Point(-1.016717f, 1.696548f, -0.914021f), Vector(0.521502f, -0.446796f, 0.726917f)));
    rays.push_back(Ray(Point(0.9999f, 1.047075f, -1.036524f), Vector(-0.269041f, -0.121707f, -0.955408f)));
    rays.push_back(Ray(Point(0.9999f, 1.979646f, -0.027152f), Vector(-0.292425f, 0.945403f, -0.143878f)));
    rays.push_back(Ray(Point(-0.878487f, 1.845481f, -1.0399f), Vector(0.934317f, 0.353708f, 0.044069f)));
    rays.push_back(Ray(Point(-0.08758f, 0.0001f, -1.024287f), Vector(-0.296119f, 0.876241f, -0.380151f)));
    rays.push_back(Ray(Point(-0.520022f, 1.9899f, -0.798637f), Vector(0.923669f, -0.199924f, 0.326903f)));
    rays.push_back(Ray(Point(0.20062f, 1.9899f, -0.839812f), Vector(0.034687f, -0.654385f, -0.755366f)));
    rays.push_back(Ray(Point(-0.823308f, 0.0001f, -0.043324f), Vector(0.939843f, 0.297356f, 0.168153f)));
    rays.push_back(Ray(Point(-1.000436f, 0.351314f, -0.508197f), Vector(0.397309f, 0.72544f, 0.562035f)));
    rays.push_back(Ray(Point(-0.848998f, 0.0001f, -0.312536f), Vector(-0.561347f, 0.761355f, 0.32439f)));
    rays.push_back(Ray(Point(-0.127585f, 1.9899f, 0.445752f), Vector(-0.036753f, -0.80982f, -0.585525f)));
    rays.push_back(Ray(Point(0.644397f, 0.0001f, -0.247642f), Vector(0.10746f, 0.475047f, 0.873374f)));
    rays.push_back(Ray(Point(0.763142f, 0.0001f, -0.569208f), Vector(0.846114f, 0.023909f, 0.532466f)));
    rays.push_back(Ray(Point(0.9999f, 1.889473f, -0.522421f), Vector(-0.32431f, 0.749641f, 0.576942f)));
    rays.push_back(Ray(Point(0.035028f, 1.031316f, -0.088325f), Vector(0.5857f, 0.804493f, 0.098728f)));
    rays.push_back(Ray(Point(0.034811f, 1.035509f, -0.088256f), Vector(0.24435f, 0.943741f, 0.222814f)));
    rays.push_back(Ray(Point(0.038867f, 1.03482f, -0.089537f), Vector(-0.664571f, -0.463351f, 0.586218f)));
    rays.push_back(Ray(Point(0.038904f, 1.038563f, -0.089549f), Vector(0.505407f, -0.767417f, 0.394506f)));
    rays.push_back(Ray(Point(0.199119f, 1.814496f, -1.0399f), Vector(0.619113f, 0.192856f, 0.761253f)));
    rays.push_back(Ray(Point(0.9999f, 1.094206f, -0.018919f), Vector(-0.825062f, -0.527125f, -0.203496f)));
    rays.push_back(Ray(Point(0.617491f, 0.053902f, 0.145288f), Vector(-0.236321f, 0.642272f, -0.729136f)));
    rays.push_back(Ray(Point(0.202887f, 1.9899f, 0.539888f), Vector(0.964917f, -0.023608f, 0.261493f)));
    rays.push_back(Ray(Point(-0.091575f, 0.011493f, -0.5143f), Vector(0.848115f, 0.529437f, -0.019909f)));
    rays.push_back(Ray(Point(0.9999f, 1.344642f, -1.015935f), Vector(-0.014165f, -0.193874f, -0.980924f)));
    rays.push_back(Ray(Point(-0.024364f, 0.325182f, -0.297731f), Vector(0.60328f, 0.714532f, 0.354256f)));
    rays.push_back(Ray(Point(0.033104f, 0.968318f, -0.087717f), Vector(0.953084f, 0.180432f, -0.243052f)));
    rays.push_back(Ray(Point(0.033475f, 0.972253f, -0.087835f), Vector(0.3399f, -0.918548f, 0.201837f)));
    rays.push_back(Ray(Point(0.0328f, 0.97436f, -0.087621f), Vector(0.998535f, 0.038155f, -0.038375f)));
    rays.push_back(Ray(Point(0.032607f, 0.984012f, -0.08756f), Vector(0.77939f, -0.228755f, 0.583286f)));
    rays.push_back(Ray(Point(0.032766f, 0.987946f, -0.08761f), Vector(-0.729107f, 0.246612f, 0.638424f)));
    rays.push_back(Ray(Point(0.037689f, 0.96658f, -0.089165f), Vector(0.458243f, -0.36959f, 0.808342f)));
    rays.push_back(Ray(Point(0.037522f, 0.970452f, -0.089112f), Vector(-0.75237f, 0.104261f, 0.650437f)));
    rays.push_back(Ray(Point(0.037195f, 0.981334f, -0.089009f), Vector(0.275179f, 0.953038f, 0.126467f)));
    rays.push_back(Ray(Point(0.036507f, 0.98744f, -0.088792f), Vector(0.882211f, 0.302554f, 0.360783f)));
    rays.push_back(Ray(Point(0.036669f, 0.9899f, -0.088843f), Vector(0.9533f, -0.037476f, -0.299691f)));
    rays.push_back(Ray(Point(0.036319f, 0.993899f, -0.088733f), Vector(0.111002f, -0.984554f, 0.135399f)));
    rays.push_back(Ray(Point(0.031661f, 1.013924f, -0.087262f), Vector(0.735594f, 0.657196f, 0.164303f)));
    rays.push_back(Ray(Point(0.031449f, 1.02048f, -0.087195f), Vector(-0.12802f, 0.809844f, 0.572507f)));
    rays.push_back(Ray(Point(0.031154f, 1.023646f, -0.087102f), Vector(0.319439f, 0.867859f, 0.380498f)));
    rays.push_back(Ray(Point(0.034989f, 1.030678f, -0.088312f), Vector(-0.796734f, -0.176284f, 0.578047f)));
    rays.push_back(Ray(Point(0.035914f, 0.998645f, -0.088605f), Vector(-0.832433f, 0.317267f, 0.454309f)));
    rays.push_back(Ray(Point(0.036142f, 1.003428f, -0.088676f), Vector(-0.593605f, 0.771377f, 0.229369f)));
    rays.push_back(Ray(Point(0.035424f, 1.013535f, -0.08845f), Vector(0.971039f, 0.14729f, 0.188119f)));
    rays.push_back(Ray(Point(0.035516f, 1.01831f, -0.088479f), Vector(0.733449f, -0.008394f, 0.679693f)));
    rays.push_back(Ray(Point(0.035231f, 1.021542f, -0.088389f), Vector(-0.745864f, -0.580055f, 0.327449f)));
    rays.push_back(Ray(Point(0.035335f, 1.025027f, -0.088422f), Vector(0.987237f, 0.032521f, 0.155904f)));
    rays.push_back(Ray(Point(0.03923f, 1.02902f, -0.089652f), Vector(0.815681f, 0.54727f, -0.187511f)));
    rays.push_back(Ray(Point(0.9999f, 0.497271f, -0.197853f), Vector(-0.814863f, -0.094057f, -0.571972f)));
    rays.push_back(Ray(Point(0.9999f, 0.076613f, 0.155382f), Vector(-0.639477f, 0.766124f, -0.064209f)));
    rays.push_back(Ray(Point(-0.639692f, 1.049121f, -0.263114f), Vector(-0.513824f, -0.462947f, 0.722264f)));
    rays.push_back(Ray(Point(0.446728f, 1.327006f, -1.0399f), Vector(-0.364148f, 0.782975f, 0.504327f)));
    rays.push_back(Ray(Point(0.182313f, 0.0001f, -0.794916f), Vector(0.909814f, 0.069553f, 0.409147f)));
    rays.push_back(Ray(Point(-1.015177f, 1.050065f, 0.373449f), Vector(0.387755f, -0.80715f, -0.445145f)));
    rays.push_back(Ray(Point(-0.625365f, 0.0001f, -0.197249f), Vector(0.645328f, 0.234375f, -0.727063f)));
    rays.push_back(Ray(Point(-1.012901f, 1.407054f, -0.858369f), Vector(0.970037f, -0.207522f, 0.126341f)));
    rays.push_back(Ray(Point(0.9999f, 0.840425f, -0.398739f), Vector(-0.156967f, -0.566705f, -0.808831f)));
    rays.push_back(Ray(Point(0.997463f, 1.9899f, 0.052148f), Vector(-0.137932f, -0.635329f, -0.759823f)));
    rays.push_back(Ray(Point(0.324973f, 0.0001f, -1.016454f), Vector(-0.839618f, 0.50697f, 0.194993f)));
    rays.push_back(Ray(Point(0.594075f, 0.0001f, -0.89986f), Vector(-0.956798f, 0.269695f, -0.108643f)));
    rays.push_back(Ray(Point(-0.623661f, 1.2001f, -0.248971f), Vector(-0.634321f, 0.046941f, -0.771644f)));
    rays.push_back(Ray(Point(-0.686717f, 1.642494f, -1.0399f), Vector(-0.191345f, -0.750909f, 0.632078f)));
    rays.push_back(Ray(Point(-1.017949f, 1.601743f, -0.504634f), Vector(0.277155f, 0.759659f, -0.588306f)));
    rays.push_back(Ray(Point(-0.484598f, 0.0001f, -1.000232f), Vector(0.086457f, 0.609417f, -0.788122f)));
    rays.push_back(Ray(Point(0.9999f, 0.763882f, -0.568534f), Vector(-0.92936f, 0.352069f, -0.111074f)));
    rays.push_back(Ray(Point(0.020099f, 1.9899f, -0.814018f), Vector(0.928768f, -0.324519f, -0.179102f)));
    rays.push_back(Ray(Point(-0.638424f, 0.175668f, -0.512708f), Vector(-0.968016f, 0.161532f, 0.19197f)));
    rays.push_back(Ray(Point(0.553125f, 0.6001f, 0.257741f), Vector(0.775182f, 0.578872f, 0.252983f)));
    rays.push_back(Ray(Point(0.502343f, 0.0001f, -0.695608f), Vector(0.006973f, 0.86923f, -0.494358f)));
    rays.push_back(Ray(Point(0.982845f, 1.817399f, -1.0399f), Vector(0.047561f, -0.696211f, 0.71626f)));
    rays.push_back(Ray(Point(0.9999f, 0.7745f, 0.20078f), Vector(-0.158906f, 0.028785f, -0.986874f)));
    rays.push_back(Ray(Point(-0.550224f, 0.0001f, 0.1191f), Vector(-0.287743f, 0.862445f, 0.416405f)));
    rays.push_back(Ray(Point(0.723359f, 0.0001f, -0.702719f), Vector(-0.133299f, 0.552568f, 0.82274f)));
    rays.push_back(Ray(Point(-0.455157f, 0.0001f, 0.551188f), Vector(0.293671f, 0.894849f, -0.336159f)));
    rays.push_back(Ray(Point(0.24247f, 0.049715f, -1.0399f), Vector(0.763494f, 0.635087f, 0.117223f)));
    rays.push_back(Ray(Point(0.732985f, 1.9899f, 0.029325f), Vector(0.288077f, -0.748485f, 0.597312f)));
    rays.push_back(Ray(Point(0.281944f, 1.9899f, 0.137096f), Vector(0.854965f, -0.330834f, -0.399479f)));
    rays.push_back(Ray(Point(0.327733f, 0.6001f, 0.135903f), Vector(-0.185744f, 0.37484f, -0.908292f)));
    rays.push_back(Ray(Point(0.977682f, 1.105059f, -1.0399f), Vector(0.135056f, -0.748458f, 0.649284f)));
    rays.push_back(Ray(Point(-1.015297f, 1.361963f, -0.546222f), Vector(0.957817f, -0.259001f, -0.124519f)));

    return rays;
}

std::vector<Point> create_ray_inter_result_points()
{
    std::vector<Point> points;

    points.push_back(Point(0.042498f, 0.6f, 0.386415f));
    points.push_back(Point(1, 0.898655f, 0.262971f));
    points.push_back(Point(1, 1.127223f, -0.16831f));
    points.push_back(Point(-0.698062f, 1.99f, 0.748508f));
    points.push_back(Point(0.018589f, 1.99f, 0.336684f));
    points.push_back(Point(-1.004969f, 0.141377f, 0.263019f));
    points.push_back(Point(-1.014911f, 0.977376f, 0.413901f));
    points.push_back(Point(1, 1.247962f, 0.754168f));
    points.push_back(Point(1, 1.778887f, 0.1815f));
    points.push_back(Point(1, 1.145532f, 0.558433f));
    points.push_back(Point(1, 0.295682f, -0.627325f));
    points.push_back(Point(1, 1.501625f, 0.816996f));
    points.push_back(Point(-0.436889f, 0.002241f, 0.060597f));
    points.push_back(Point(1, 0.136332f, -0.394297f));
    points.push_back(Point(0.576406f, 0.385868f, 0.591675f));
    points.push_back(Point(0.409619f, 0.755624f, -1.04f));
    points.push_back(Point(0.720363f, 1.348355f, -1.04f));
    points.push_back(Point(0.166112f, 0.236172f, -1.04f));
    points.push_back(Point(1, 1.326105f, -0.927948f));
    points.push_back(Point(-1.010209f, 0.934278f, -0.418358f));
    points.push_back(Point(0.987602f, 0.330698f, -1.04f));
    points.push_back(Point(0.648027f, 0.351353f, 0.154499f));
    points.push_back(Point(1, 0.781552f, -0.94911f));
    points.push_back(Point(-0.592815f, 0.24825f, -0.112403f));
    points.push_back(Point(-0.511216f, 1.99f, 0.669427f));
    points.push_back(Point(-1.016307f, 1.255106f, 0.517779f));
    points.push_back(Point(-0.034441f, 0.120908f, 0.52073f));
    points.push_back(Point(-1.019896f, 1.969337f, -0.574504f));
    points.push_back(Point(-0.311951f, 1.99f, 0.609596f));
    points.push_back(Point(1, 1.536174f, -0.179516f));
    points.push_back(Point(0.291028f, 0.6f, 0.475652f));
    points.push_back(Point(-0.970465f, 1.99f, 0.435633f));
    points.push_back(Point(0.837753f, 1.99f, -0.057874f));
    points.push_back(Point(-1.014568f, 0.908996f, 0.451714f));
    points.push_back(Point(1, 1.07791f, 0.069639f));
    points.push_back(Point(1, 1.813419f, 0.552326f));
    points.push_back(Point(0.57689f, 0.6f, 0.219942f));
    points.push_back(Point(1, 1.184152f, -0.296873f));
    points.push_back(Point(0.226008f, 0.6f, 0.20114f));
    points.push_back(Point(0.035342f, 1.99f, 0.671371f));
    points.push_back(Point(1, 0.355452f, -0.285097f));
    points.push_back(Point(0.807638f, 1.99f, -0.31788f));
    points.push_back(Point(-1.01496f, 0.987026f, 0.646362f));
    points.push_back(Point(1, 1.246021f, 0.628447f));
    points.push_back(Point(-1.007954f, 0.161804f, 0.534782f));
    points.push_back(Point(-0.506683f, 1.99f, 0.620413f));
    points.push_back(Point(1, 1.212836f, 0.211835f));
    points.push_back(Point(1, 0.570557f, 0.585686f));
    points.push_back(Point(0.379932f, 1.99f, -0.04837f));
    points.push_back(Point(1, 0.542911f, 0.509043f));
    points.push_back(Point(-1.019518f, 1.894077f, 0.317029f));
    points.push_back(Point(-0.967159f, 1.99f, 0.819625f));
    points.push_back(Point(1, 1.588232f, 0.557971f));
    points.push_back(Point(1, 0.518023f, 0.13489f));
    points.push_back(Point(0.09776f, 0.335115f, 0.102092f));
    points.push_back(Point(-0.596933f, 1.99f, 0.670822f));
    points.push_back(Point(0.569661f, 0.6f, 0.17942f));
    points.push_back(Point(1, 0.929353f, 0.206319f));
    points.push_back(Point(1, 1.345194f, -0.047569f));
    points.push_back(Point(0.902787f, 1.99f, 0.339958f));
    points.push_back(Point(0.71214f, 1.99f, 0.742275f));
    points.push_back(Point(0.250132f, 0.6f, 0.378377f));
    points.push_back(Point(0.35583f, 1.99f, 0.042329f));
    points.push_back(Point(1, 0.664979f, -0.2557f));
    points.push_back(Point(-1.01598f, 1.19006f, 0.881161f));
    points.push_back(Point(1, 0.052446f, -0.278826f));
    points.push_back(Point(-1.017126f, 1.418095f, 0.643108f));
    points.push_back(Point(-0.072237f, 1.99f, 0.220521f));
    points.push_back(Point(0.049192f, 0.6f, 0.463426f));
    points.push_back(Point(1, 1.710772f, -0.098702f));
    points.push_back(Point(0.449549f, 0.6f, 0.274145f));
    points.push_back(Point(0.56545f, 1.99f, -0.071923f));
    points.push_back(Point(1, 0.302848f, 0.176226f));
    points.push_back(Point(1, 1.662994f, -0.371278f));
    points.push_back(Point(0.03336f, 1.99f, 0.172843f));
    points.push_back(Point(-1.01207f, 0.411882f, 0.783593f));
    points.push_back(Point(-0.356008f, 1.99f, 0.613468f));
    points.push_back(Point(0.331174f, 0.6f, 0.129577f));
    points.push_back(Point(0.422289f, 1.99f, 0.163647f));
    points.push_back(Point(0.80837f, 1.99f, 0.277665f));
    points.push_back(Point(1, 0.828617f, 0.089051f));
    points.push_back(Point(1, 0.48617f, -0.44101f));
    points.push_back(Point(0.882515f, 1.99f, -0.124883f));
    points.push_back(Point(0.94745f, 1.266768f, -1.04f));
    points.push_back(Point(-0.08723f, 1.99f, 0.575471f));
    points.push_back(Point(1, 1.23877f, 0.615227f));
    points.push_back(Point(0.121435f, 0.6f, 0.151576f));
    points.push_back(Point(1, 1.305636f, 0.264663f));
    points.push_back(Point(-1.013806f, 0.757368f, 0.424118f));
    points.push_back(Point(1, 0.939995f, -0.362141f));
    points.push_back(Point(-1.012905f, 0.578053f, 0.871119f));
    points.push_back(Point(-1.017566f, 1.505624f, 0.298463f));
    points.push_back(Point(-0.315754f, 1.99f, 0.066318f));
    points.push_back(Point(1, 1.799586f, 0.469271f));
    points.push_back(Point(1, 1.730928f, 0.177543f));
    points.push_back(Point(-1.015288f, 1.052379f, 0.454325f));
    points.push_back(Point(1, 0.850421f, 0.145165f));
    points.push_back(Point(0.566487f, 0.6f, 0.288491f));
    points.push_back(Point(-1.01799f, 1.589976f, 0.536692f));
    points.push_back(Point(-1.016605f, 1.314381f, 0.360997f));
    points.push_back(Point(-1.01246f, 0.48961f, 0.959613f));
    points.push_back(Point(-1.014694f, 0.934159f, 0.960433f));
    points.push_back(Point(0.185121f, 0.6f, 0.226484f));
    points.push_back(Point(1, 0.202638f, -0.128422f));
    points.push_back(Point(1, 1.242888f, -0.367192f));
    points.push_back(Point(0.099134f, 0.6f, 0.331814f));
    points.push_back(Point(-0.060222f, 1.98f, 0.062737f));
    points.push_back(Point(1, 1.297075f, 0.362705f));
    points.push_back(Point(0.923828f, 1.99f, -0.087266f));
    points.push_back(Point(1, 1.451276f, 0.744477f));
    points.push_back(Point(0.474502f, 0.6f, 0.154897f));
    points.push_back(Point(0.554944f, 1.99f, -0.240485f));
    points.push_back(Point(0.048103f, 0.6f, 0.356522f));
    points.push_back(Point(0.405195f, 0.6f, 0.348554f));
    points.push_back(Point(0.399675f, 0.6f, 0.613953f));
    points.push_back(Point(1, 1.497201f, 0.600343f));
    points.push_back(Point(-0.301804f, 1.99f, 0.337268f));
    points.push_back(Point(-0.054778f, 1.99f, 0.956846f));
    points.push_back(Point(0.358598f, 0.6f, 0.37706f));
    points.push_back(Point(0.559591f, 1.99f, -0.072092f));
    points.push_back(Point(0.797666f, 1.99f, 0.403953f));
    points.push_back(Point(-0.597958f, 1.99f, 0.521849f));
    points.push_back(Point(1, 1.623055f, -0.069328f));
    points.push_back(Point(1, 0.827299f, 0.020994f));
    points.push_back(Point(0.573324f, 0.6f, 0.172398f));
    points.push_back(Point(0.019224f, 1.99f, 0.204016f));
    points.push_back(Point(-1.014059f, 0.80781f, 0.247459f));
    points.push_back(Point(-0.028295f, 0.302521f, 0.501266f));
    points.push_back(Point(-1.018394f, 1.670467f, -0.38636f));
    points.push_back(Point(-1.01979f, 1.948197f, 0.417555f));
    points.push_back(Point(0.061988f, 1.99f, 0.794787f));
    points.push_back(Point(0.144977f, 0.217685f, 0.63051f));
    points.push_back(Point(0.100104f, 0.255225f, 0.094669f));
    points.push_back(Point(1, 1.941455f, 0.600066f));
    points.push_back(Point(-1.018321f, 1.655825f, 0.482316f));
    points.push_back(Point(1, 0.00303f, -0.020174f));
    points.push_back(Point(0.927658f, 0.943135f, -1.04f));
    points.push_back(Point(1, 0.344955f, 0.152304f));
    points.push_back(Point(0.837642f, 1.99f, 0.060962f));
    points.push_back(Point(0.792053f, 1.99f, -0.75893f));
    points.push_back(Point(-1.018516f, 1.694662f, 0.792401f));
    points.push_back(Point(1, 1.121728f, -0.073316f));
    points.push_back(Point(-1.017776f, 1.547436f, 0.653346f));
    points.push_back(Point(-0.487702f, 1.99f, -0.66509f));
    points.push_back(Point(-0.059473f, 0.967066f, -0.410525f));
    points.push_back(Point(-0.959996f, 1.99f, -0.124791f));
    points.push_back(Point(-0.325264f, 1.99f, -0.251423f));
    points.push_back(Point(1, 0.255768f, 0.911092f));
    points.push_back(Point(0.333328f, 1.99f, 0.528871f));
    points.push_back(Point(-1.011215f, 0.241724f, 0.933914f));
    points.push_back(Point(0.290514f, 1.99f, 0.735056f));
    points.push_back(Point(0.178353f, 0.322839f, -1.04f));
    points.push_back(Point(1, 1.857758f, -0.138231f));
    points.push_back(Point(-0.865242f, 1.99f, 0.458778f));
    points.push_back(Point(0.563589f, 1.99f, -0.412589f));
    points.push_back(Point(-1.01574f, 1.142195f, 0.147094f));
    points.push_back(Point(-1.016879f, 1.368841f, 0.575301f));
    points.push_back(Point(0.416243f, 0.6f, 0.544809f));
    points.push_back(Point(-1.009738f, 0.247243f, 0.585104f));
    points.push_back(Point(1, 0.55367f, 0.541796f));
    points.push_back(Point(0.837767f, 1.99f, 0.793623f));
    points.push_back(Point(-0.036512f, 0.144761f, 0.527288f));
    points.push_back(Point(-0.295319f, 0.254636f, -1.04f));
    points.push_back(Point(-1.013308f, 0.65831f, 0.468006f));
    points.push_back(Point(0.321691f, 0.6f, 0.158978f));
    points.push_back(Point(1, 1.354367f, 0.444087f));
    points.push_back(Point(0.34599f, 1.606725f, -1.04f));
    points.push_back(Point(1, 0.85329f, 0.031739f));
    points.push_back(Point(1, 1.920189f, 0.708497f));
    points.push_back(Point(0.022471f, 0.625893f, -0.146482f));
    points.push_back(Point(-0.992537f, 0.364896f, -1.04f));
    points.push_back(Point(-1.019244f, 1.839649f, 0.108427f));
    points.push_back(Point(-0.748619f, 1.99f, -0.200296f));
    points.push_back(Point(-1.008212f, 1.99f, 0.425102f));
    points.push_back(Point(-0.026607f, 1.99f, 0.850785f));
    points.push_back(Point(-0.252336f, 1.99f, 0.577714f));
    points.push_back(Point(0.549229f, 1.99f, -0.162089f));
    points.push_back(Point(-1.017798f, 1.671045f, -0.775468f));
    points.push_back(Point(0.601092f, 1.99f, 0.894116f));
    points.push_back(Point(1, 1.74584f, 0.098417f));
    points.push_back(Point(1, 1.869018f, -0.864785f));
    points.push_back(Point(1, 0.06999f, -0.343425f));
    points.push_back(Point(-0.966139f, 1.99f, 0.554426f));
    points.push_back(Point(1, 0.854463f, 0.359756f));
    points.push_back(Point(-1.018068f, 1.605479f, -0.569592f));
    points.push_back(Point(0.348965f, 1.201217f, -1.04f));
    points.push_back(Point(-1.019761f, 1.942384f, -0.126035f));
    points.push_back(Point(1, 0.029142f, -0.205608f));
    points.push_back(Point(-0.750382f, 1.884639f, -1.04f));
    points.push_back(Point(-1.01046f, 0.450578f, 0.347273f));
    points.push_back(Point(-1.010268f, 0.120123f, 0.83341f));
    points.push_back(Point(1, 0.025702f, -0.200312f));
    points.push_back(Point(-0.585112f, 0.333553f, -1.04f));
    points.push_back(Point(1, 1.027484f, -0.225799f));
    points.push_back(Point(0.486772f, 1.99f, -0.942847f));
    points.push_back(Point(0.393837f, 0.29724f, 0.078688f));
    points.push_back(Point(0.052119f, 1.99f, 0.793421f));
    points.push_back(Point(1, 1.280513f, 0.105702f));
    points.push_back(Point(1, 1.607081f, 0.242348f));
    points.push_back(Point(0.091383f, 0.257273f, 0.122288f));
    points.push_back(Point(-0.162568f, 1.788709f, -1.04f));
    points.push_back(Point(1, 1.474964f, 0.747578f));
    points.push_back(Point(-0.563395f, 1.99f, 0.915257f));
    points.push_back(Point(0.285035f, 0.6f, 0.220832f));
    points.push_back(Point(1, 1.033705f, -0.018579f));
    points.push_back(Point(1, 0.585279f, 0.284616f));
    points.push_back(Point(0.881483f, 1.99f, 0.818708f));
    points.push_back(Point(0.294915f, 1.99f, 0.183781f));
    points.push_back(Point(-1.014249f, 0.845559f, 0.502726f));
    points.push_back(Point(-0.039539f, 1.98f, 0.138218f));
    points.push_back(Point(0.548451f, 0.6f, 0.18039f));
    points.push_back(Point(1, 0.398487f, 0.06048f));
    points.push_back(Point(1, 0.421978f, -0.114174f));
    points.push_back(Point(1, 1.658244f, -0.033258f));
    points.push_back(Point(-0.229472f, 1.99f, 0.565577f));
    points.push_back(Point(0.967153f, 1.99f, 0.508981f));
    points.push_back(Point(-1.004968f, 0.063906f, 0.381509f));
    points.push_back(Point(1, 1.596304f, -0.184391f));
    points.push_back(Point(0.317208f, 0.6f, 0.296303f));
    points.push_back(Point(0.47731f, 0.6f, 0.209325f));
    points.push_back(Point(0.524842f, 0.6f, 0.193247f));
    points.push_back(Point(0.652238f, 0.6f, 0.179219f));
    points.push_back(Point(0.24435f, 0.6f, 0.038352f));
    points.push_back(Point(0.626319f, 1.99f, 0.832438f));
    points.push_back(Point(-1.017194f, 1.431592f, 0.312991f));
    points.push_back(Point(1, 1.382225f, -0.367423f));
    points.push_back(Point(1, 1.314756f, 0.644446f));
    points.push_back(Point(1, 1.906041f, 0.463451f));
    points.push_back(Point(-0.520205f, 1.99f, 0.829438f));
    points.push_back(Point(0.479743f, 1.99f, -0.123462f));
    points.push_back(Point(-0.484816f, 1.99f, 0.129553f));
    points.push_back(Point(0.336718f, 1.99f, -0.118113f));
    points.push_back(Point(1, 0.661993f, 0.259911f));
    points.push_back(Point(0.392089f, 1.99f, 0.623304f));
    points.push_back(Point(1, 1.215727f, 0.834748f));
    points.push_back(Point(0.047354f, 0.427792f, 0.261712f));
    points.push_back(Point(0.164903f, 0.6f, 0.013379f));
    points.push_back(Point(-0.223458f, 1.99f, 0.872102f));
    points.push_back(Point(1, 0.964592f, -0.265562f));
    points.push_back(Point(0.490414f, 1.99f, 0.834645f));
    points.push_back(Point(0.221586f, 0.6f, 0.127118f));
    points.push_back(Point(1, 0.540271f, 0.305003f));
    points.push_back(Point(1, 1.009351f, -0.238167f));
    points.push_back(Point(1, 1.146473f, -0.042495f));
    points.push_back(Point(1, 1.337233f, 0.669834f));
    points.push_back(Point(-0.874042f, 1.841374f, -1.04f));
    points.push_back(Point(0.529252f, 1.99f, -0.452267f));
    points.push_back(Point(-1.019576f, 1.905672f, -0.31868f));
    points.push_back(Point(-0.351315f, 1.99f, 0.155756f));
    points.push_back(Point(1, 1.967628f, 0.05312f));
    points.push_back(Point(1, 0.666454f, -0.342149f));
    points.push_back(Point(1, 1.260335f, 0.419643f));
    points.push_back(Point(1, 1.343159f, 0.911459f));
    points.push_back(Point(-1.007871f, 0.399797f, 0.162119f));
    points.push_back(Point(1, 1.411455f, 0.322277f));
    points.push_back(Point(0.277599f, 1.045296f, -1.04f));
    points.push_back(Point(-0.342639f, 0.568736f, 0.030834f));
    points.push_back(Point(-0.08747f, 1.105258f, -1.04f));
    points.push_back(Point(-0.651665f, 0.409658f, -1.04f));
    points.push_back(Point(-0.341718f, 0.807941f, 0.030542f));
    points.push_back(Point(-0.644575f, 0.546994f, -0.279186f));
    points.push_back(Point(-0.065681f, 0.644043f, -0.430528f));
    points.push_back(Point(-0.941846f, 1.99f, 0.618139f));
    points.push_back(Point(-1.01881f, 1.753283f, -0.378595f));
    points.push_back(Point(-0.0385f, 0.794797f, -0.06521f));
    points.push_back(Point(0.07723f, 1.150287f, -1.04f));
    points.push_back(Point(-0.527796f, 0.106505f, -1.04f));
    points.push_back(Point(1, 0.64935f, 0.629664f));
    points.push_back(Point(0.458474f, 1.122784f, -1.04f));
    points.push_back(Point(1, 1.093782f, 0.054299f));
    points.push_back(Point(1, 0.794155f, 0.497215f));
    points.push_back(Point(1, 0.907201f, -0.95714f));
    points.push_back(Point(-0.410453f, 1.2f, 0.028917f));
    points.push_back(Point(-0.615467f, 0.15464f, -1.04f));
    points.push_back(Point(-0.995553f, 0.366864f, -1.037751f));
    points.push_back(Point(0.636359f, 0.057102f, 0.387128f));
    points.push_back(Point(0.127021f, 1.99f, 0.484577f));
    points.push_back(Point(0.325465f, 0.371475f, 0.058297f));
    points.push_back(Point(-0.332301f, 1.076544f, -0.609273f));
    points.push_back(Point(1, 1.528304f, 0.678035f));
    points.push_back(Point(-1.018995f, 1.790055f, -0.044683f));
    points.push_back(Point(0.960148f, 1.880736f, -1.04f));
    points.push_back(Point(0.067739f, 1.521068f, -1.04f));
    points.push_back(Point(-0.87456f, 1.99f, -0.194758f));
    points.push_back(Point(-1.015437f, 1.61135f, -0.923785f));
    points.push_back(Point(0.670162f, 1.360363f, -1.04f));
    points.push_back(Point(1, 0.413289f, -0.133218f));
    points.push_back(Point(0.716299f, 1.99f, -0.076666f));
    points.push_back(Point(0.012682f, 0.018167f, -0.081373f));
    points.push_back(Point(-1.019058f, 1.802561f, -0.837609f));
    points.push_back(Point(0.309648f, 0.6f, 0.375605f));
    points.push_back(Point(0.186336f, 1.99f, 0.717227f));
    points.push_back(Point(-1.019706f, 1.931574f, 0.619043f));
    points.push_back(Point(0.021833f, 0.158909f, 0.34253f));
    points.push_back(Point(-0.157835f, 1.99f, 0.901284f));
    points.push_back(Point(-0.303797f, 1.99f, 0.310034f));
    points.push_back(Point(1, 1.223758f, -0.355869f));
    points.push_back(Point(1, 1.046063f, 0.20077f));
    points.push_back(Point(1, 0.929343f, -0.219553f));
    points.push_back(Point(0.351242f, 1.99f, 0.63562f));
    points.push_back(Point(1, 0.87903f, -0.140403f));
    points.push_back(Point(-1.012294f, 0.456421f, 0.82974f));
    points.push_back(Point(1, 1.133146f, -1.017336f));
    points.push_back(Point(0.966626f, 0.55211f, -1.04f));
    points.push_back(Point(-1.013838f, 0.763744f, 0.498959f));
    points.push_back(Point(1, 1.167192f, 0.152746f));
    points.push_back(Point(-1.014767f, 0.948643f, 0.713242f));
    points.push_back(Point(0.27254f, 1.99f, -0.071222f));
    points.push_back(Point(0.889821f, 1.99f, 0.643035f));
    points.push_back(Point(-0.767061f, 1.99f, 0.169389f));
    points.push_back(Point(1, 1.868887f, -0.189269f));
    points.push_back(Point(1, 0.699685f, 0.385696f));
    points.push_back(Point(0.316975f, 0.6f, 0.456913f));
    points.push_back(Point(1, 0.136768f, -0.130801f));
    points.push_back(Point(1, 1.118179f, -0.021146f));
    points.push_back(Point(0.31543f, 0.6f, 0.10868f));
    points.push_back(Point(-1.017415f, 1.47553f, 0.836071f));
    points.push_back(Point(1, 1.393942f, -0.196892f));
    points.push_back(Point(-1.012394f, 0.476336f, 0.74864f));
    points.push_back(Point(1, 1.929564f, 0.098141f));
    points.push_back(Point(1, 0.95493f, -0.366595f));
    points.push_back(Point(1, 1.579473f, 0.253663f));
    points.push_back(Point(0.246778f, 0.6f, 0.477758f));
    points.push_back(Point(-1.013369f, 0.670538f, 0.426898f));
    points.push_back(Point(1, 1.682019f, -0.25468f));
    points.push_back(Point(0.692004f, 1.99f, -0.168413f));
    points.push_back(Point(0.379683f, 0.6f, 0.330765f));
    points.push_back(Point(1, 0.634218f, 0.322396f));
    points.push_back(Point(-0.519074f, 1.99f, 0.093457f));
    points.push_back(Point(1, 1.719544f, -0.312502f));
    points.push_back(Point(1, 0.858024f, -0.198428f));
    points.push_back(Point(1, 1.299146f, -0.010365f));
    points.push_back(Point(0.423196f, 1.99f, 0.285916f));
    points.push_back(Point(1, 1.207669f, 0.431329f));
    points.push_back(Point(0.304645f, 0.6f, 0.392584f));
    points.push_back(Point(-0.147973f, 0.868808f, -0.03064f));
    points.push_back(Point(-0.077935f, 0.331392f, -0.052757f));
    points.push_back(Point(-0.152008f, 0.756405f, -0.029366f));
    points.push_back(Point(-0.57006f, 1.2f, -0.27711f));
    points.push_back(Point(1, 0.850112f, 0.078524f));
    points.push_back(Point(0.615192f, 1.99f, -0.078757f));
    points.push_back(Point(-0.262308f, 1.198156f, -1.04f));
    points.push_back(Point(-1.01628f, 1.249667f, 0.03153f));
    points.push_back(Point(0.807612f, 1.99f, -0.249529f));
    points.push_back(Point(-1.012129f, 0.423595f, 0.895762f));
    points.push_back(Point(1, 1.227964f, -0.882933f));
    points.push_back(Point(-1.015017f, 0.998457f, 0.015282f));
    points.push_back(Point(-0.259207f, 1.99f, -0.125108f));
    points.push_back(Point(-0.115106f, 0.163604f, -1.04f));
    points.push_back(Point(-0.415586f, 1.300735f, -1.04f));
    points.push_back(Point(-1.010243f, 0.476301f, 0.28579f));
    points.push_back(Point(1, 0.2427f, -0.063842f));
    points.push_back(Point(0.528143f, 1.99f, 0.06674f));
    points.push_back(Point(-0.120372f, 1.636137f, -1.04f));
    points.push_back(Point(-0.018432f, 1.082918f, -0.27828f));
    points.push_back(Point(0.940738f, 0.898469f, -1.04f));
    points.push_back(Point(0.792971f, 1.99f, -0.034748f));
    points.push_back(Point(1, 0.727313f, -0.349072f));
    points.push_back(Point(-1.002605f, 0.438116f, -0.430934f));
    points.push_back(Point(-1.019575f, 1.905502f, 0.942847f));
    points.push_back(Point(-0.604973f, 0.41085f, -0.15158f));
    points.push_back(Point(-1.013283f, 0.653269f, 0.650636f));
    points.push_back(Point(0.370941f, 0.6f, 0.399191f));
    points.push_back(Point(0.452691f, 1.99f, 0.720107f));
    points.push_back(Point(1, 0.714179f, 0.080643f));
    points.push_back(Point(-1.018494f, 1.690298f, 0.307947f));
    points.push_back(Point(1, 0.42318f, 0.145944f));
    points.push_back(Point(0.073762f, 1.98f, 0.016963f));
    points.push_back(Point(0.186633f, 0.6f, 0.107405f));
    points.push_back(Point(-0.490763f, 1.99f, 0.7869f));
    points.push_back(Point(1, 0.884174f, -0.267955f));
    points.push_back(Point(0.346029f, 1.99f, 0.434608f));
    points.push_back(Point(1, 1.300859f, 0.205213f));
    points.push_back(Point(0.480921f, 0.6f, 0.185466f));
    points.push_back(Point(-0.614127f, 1.99f, 0.830581f));
    points.push_back(Point(-1.015884f, 1.170897f, 0.83952f));
    points.push_back(Point(1, 1.342215f, 0.466224f));
    points.push_back(Point(0.81508f, 1.99f, 0.731699f));
    points.push_back(Point(-1.012158f, 0.455925f, 0.511425f));
    points.push_back(Point(-0.408844f, 1.99f, 0.223544f));
    points.push_back(Point(1, 0.606556f, 0.654143f));
    points.push_back(Point(-0.061717f, 0.339666f, -0.057879f));
    points.push_back(Point(1, 1.072494f, -0.169708f));
    points.push_back(Point(0.026905f, 0.637154f, -0.085865f));
    points.push_back(Point(0.910856f, 1.719405f, -1.04f));
    points.push_back(Point(0.482328f, 0.445861f, -1.04f));
    points.push_back(Point(0.955844f, 0.369212f, -1.04f));
    points.push_back(Point(1, 1.889473f, -0.522421f));
    points.push_back(Point(0.623506f, 1.99f, -0.440387f));
    points.push_back(Point(0.171303f, 0.6f, 0.381f));
    points.push_back(Point(1, 0.487624f, -0.21139f));
    points.push_back(Point(-0.688038f, 1.99f, -0.498795f));
    points.push_back(Point(0.460414f, 1.99f, 0.748285f));
    points.push_back(Point(1, 1.815607f, -0.033645f));
    points.push_back(Point(0.404285f, 0.122949f, 0.081804f));
    points.push_back(Point(-0.769957f, 1.99f, 0.985279f));
    points.push_back(Point(1, 0.497271f, -0.197853f));
    points.push_back(Point(1, 0.076613f, 0.155382f));
    points.push_back(Point(-0.961448f, 1.99f, -0.260419f));
    points.push_back(Point(-0.06015f, 0.344158f, -0.412705f));
    points.push_back(Point(-1.016817f, 1.696547f, -0.914022f));
    points.push_back(Point(1, 1.047075f, -1.036524f));
    points.push_back(Point(-0.203045f, 0.288908f, -0.013249f));
    points.push_back(Point(0.20062f, 1.99f, -0.839812f));
    points.push_back(Point(-1.000536f, 0.351313f, -0.508198f));
    points.push_back(Point(-0.127585f, 1.99f, 0.445752f));
    points.push_back(Point(1, 1.979646f, -0.027152f));
    points.push_back(Point(0.022436f, 0.00746f, -0.146594f));
    points.push_back(Point(0.025562f, 0.051804f, 0.330722f));
    points.push_back(Point(0.954602f, 1.236945f, -1.04f));
    points.push_back(Point(1, 0.005848f, 0.045062f));
    points.push_back(Point(-0.878487f, 1.845481f, -1.04f));
    points.push_back(Point(-0.520022f, 1.99f, -0.798637f));
    points.push_back(Point(-1.013001f, 1.407053f, -0.85837f));
    points.push_back(Point(-0.481329f, 1.139742f, -1.04f));
    points.push_back(Point(1, 0.840425f, -0.398739f));
    points.push_back(Point(0.126986f, 0.271316f, 0.009545f));
    points.push_back(Point(-1.015277f, 1.050065f, 0.373449f));
    points.push_back(Point(-0.623661f, 1.2f, -0.248971f));
    points.push_back(Point(-0.686717f, 1.642494f, -1.04f));
    points.push_back(Point(-1.018049f, 1.601743f, -0.504634f));
    points.push_back(Point(0.532109f, 0.11818f, 0.742804f));
    points.push_back(Point(0.997463f, 1.99f, 0.052148f));
    points.push_back(Point(0.553125f, 0.6f, 0.257741f));
    points.push_back(Point(0.982845f, 1.817399f, -1.04f));
    points.push_back(Point(-0.639596f, 1.049121f, -0.263143f));
    points.push_back(Point(0.446728f, 1.327006f, -1.04f));
    points.push_back(Point(1, 0.763882f, -0.568534f));
    points.push_back(Point(1, 0.003023f, -0.325987f));
    points.push_back(Point(0.020099f, 1.99f, -0.814018f));
    points.push_back(Point(-0.638394f, 0.175668f, -0.512613f));
    points.push_back(Point(0.327701f, 0.243687f, -1.04f));
    points.push_back(Point(1, 1.43572f, -0.434972f));
    points.push_back(Point(-0.998744f, 0.525706f, -0.95689f));
    points.push_back(Point(1, 0.7745f, 0.20078f));
    points.push_back(Point(-0.972423f, 1.99f, 0.646228f));
    points.push_back(Point(0.24247f, 0.049715f, -1.04f));
    points.push_back(Point(0.199119f, 1.814496f, -1.04f));
    points.push_back(Point(1, 1.094206f, -0.018919f));
    points.push_back(Point(0.648323f, 0.534978f, 0.346308f));
    points.push_back(Point(0.617462f, 0.053902f, 0.145383f));
    points.push_back(Point(-1.014379f, 0.87139f, 0.752203f));
    points.push_back(Point(-1.007297f, 0.304861f, 0.249179f));
    points.push_back(Point(0.202887f, 1.99f, 0.539888f));
    points.push_back(Point(-0.09167f, 0.011493f, -0.51427f));
    points.push_back(Point(1, 1.344642f, -1.015935f));
    points.push_back(Point(-0.024459f, 0.325182f, -0.297702f));
    points.push_back(Point(0.795824f, 1.559771f, -1.04f));
    points.push_back(Point(0.949892f, 0.828884f, -1.04f));
    points.push_back(Point(0.977682f, 1.105059f, -1.04f));
    points.push_back(Point(0.128536f, 0.183492f, 0.004637f));
    points.push_back(Point(-1.015397f, 1.361961f, -0.546224f));
    points.push_back(Point(0.951157f, 1.641352f, -1.04f));
    points.push_back(Point(0.818315f, 1.99f, -0.102109f));
    points.push_back(Point(0.924601f, 1.99f, 0.387681f));
    points.push_back(Point(-1.013315f, 0.659798f, 0.891994f));
    points.push_back(Point(0.049067f, 0.332971f, 0.256287f));
    points.push_back(Point(0.174049f, 1.99f, 0.615759f));
    points.push_back(Point(0.655943f, 0.442018f, 0.32031f));
    points.push_back(Point(0.966045f, 1.99f, -0.479379f));
    points.push_back(Point(-0.316967f, 0.403011f, 0.022726f));
    points.push_back(Point(-0.31604f, 1.99f, 0.660354f));
    points.push_back(Point(-1.012613f, 0.520049f, 0.811239f));
    points.push_back(Point(0.795322f, 0.366083f, -1.04f));
    points.push_back(Point(0.469164f, 0.6f, 0.674555f));
    points.push_back(Point(-0.708115f, 1.99f, 0.929039f));
    points.push_back(Point(0.236449f, 0.704917f, -1.04f));
    points.push_back(Point(-1.019624f, 1.915273f, -0.612668f));
    points.push_back(Point(-0.152615f, 1.063744f, -0.029174f));
    points.push_back(Point(0.980851f, 1.99f, -0.163821f));
    points.push_back(Point(-1.019184f, 1.827717f, 0.61346f));
    points.push_back(Point(-1.01173f, 0.344238f, 0.911934f));
    points.push_back(Point(-1.009962f, 0.700316f, -0.0854f));
    points.push_back(Point(-1.015316f, 1.05785f, 0.077771f));
    points.push_back(Point(-1.013364f, 0.669518f, 0.840785f));
    points.push_back(Point(0.997846f, 1.99f, -0.532013f));
    points.push_back(Point(1, 1.391769f, 0.376161f));
    points.push_back(Point(0.002645f, 0.001899f, 0.403292f));
    points.push_back(Point(0.618465f, 0.00976f, 0.448179f));
    points.push_back(Point(1, 1.748562f, -0.206783f));
    points.push_back(Point(-1.018027f, 1.597394f, 0.186491f));
    points.push_back(Point(-0.42238f, 1.99f, 0.987564f));
    points.push_back(Point(1, 1.064214f, 0.773441f));
    points.push_back(Point(0.179529f, 0.6f, 0.434863f));
    points.push_back(Point(1, 1.104419f, 0.684943f));
    points.push_back(Point(1, 0.803653f, 0.226956f));
    points.push_back(Point(-0.616188f, 1.690879f, -1.04f));
    points.push_back(Point(-1.015265f, 1.421219f, -0.650249f));
    points.push_back(Point(-0.425293f, 0.112378f, 0.056935f));
    points.push_back(Point(-1.011265f, 0.508516f, 0.340282f));
    points.push_back(Point(-1.016391f, 1.271893f, 0.368953f));
    points.push_back(Point(0.053281f, 1.887468f, -1.04f));
    points.push_back(Point(-0.905906f, 1.680138f, -1.04f));
    points.push_back(Point(-0.437143f, 1.2f, -0.106158f));
    points.push_back(Point(0.998921f, 1.046632f, -1.04f));
    points.push_back(Point(0.996697f, 1.99f, -0.028727f));
    points.push_back(Point(-0.496742f, 1.99f, -1.021894f));
    points.push_back(Point(-0.09982f, 0.036319f, -1.04f));
    points.push_back(Point(1, 1.660899f, -0.260674f));
    points.push_back(Point(0.209812f, 1.816474f, -1.04f));
    points.push_back(Point(-0.556565f, 0.084495f, 0.004401f));
    points.push_back(Point(-0.10296f, 1.99f, 0.761376f));
    points.push_back(Point(-1.001146f, 0.206459f, -0.224613f));
    points.push_back(Point(-0.163434f, 1.2f, -0.12537f));
    points.push_back(Point(0.695623f, 0.226554f, 0.168695f));
    points.push_back(Point(1, 0.006793f, -0.420151f));
    points.push_back(Point(0.95641f, 1.99f, -0.445053f));
    points.push_back(Point(0.732985f, 1.99f, 0.029325f));
    points.push_back(Point(0.281944f, 1.99f, 0.137096f));
    points.push_back(Point(-1.01152f, 0.30247f, 0.837009f));
    points.push_back(Point(0.327733f, 0.6f, 0.135903f));
    points.push_back(Point(0.762528f, 1.99f, -0.347139f));
    points.push_back(Point(-0.015677f, 0.445362f, -0.269405f));
    points.push_back(Point(0.233325f, 1.097983f, -1.04f));
    points.push_back(Point(1, 1.970397f, 0.755907f));
    points.push_back(Point(1, 0.69291f, -0.539924f));
    points.push_back(Point(0.999553f, 1.339885f, -1.04f));
    points.push_back(Point(1, 1.538452f, 0.303792f));
    points.push_back(Point(1, 1.151365f, -0.334291f));
    points.push_back(Point(0.233249f, 0.432385f, 0.030793f));
    points.push_back(Point(1, 1.011317f, -0.124792f));
    points.push_back(Point(1, 0.700077f, 0.636425f));
    points.push_back(Point(-1.016748f, 1.342932f, 0.831369f));
    points.push_back(Point(0.4922f, 0.6f, 0.712594f));
    points.push_back(Point(-1.01561f, 1.116392f, 0.821339f));
    points.push_back(Point(0.328436f, 1.99f, 0.04484f));
    points.push_back(Point(1, 1.31787f, 0.305231f));
    points.push_back(Point(1, 0.952029f, -0.391688f));
    points.push_back(Point(0.124178f, 0.214621f, 0.018436f));
    points.push_back(Point(1, 1.87906f, 0.129028f));
    points.push_back(Point(-0.121812f, 1.99f, 0.598193f));
    points.push_back(Point(0.386848f, 1.99f, 0.33658f));
    points.push_back(Point(-1.014013f, 0.798578f, 0.672761f));
    points.push_back(Point(-1.017035f, 1.399958f, 0.486053f));
    points.push_back(Point(-0.723064f, 1.99f, 0.20468f));
    points.push_back(Point(1, 1.159845f, 0.098417f));
    points.push_back(Point(1, 1.007271f, 0.805317f));
    points.push_back(Point(-1.007064f, 0.210955f, 0.369199f));
    points.push_back(Point(1, 1.056804f, 0.063918f));
    points.push_back(Point(1, 1.673635f, -0.310516f));
    points.push_back(Point(-0.199869f, 0.358785f, -1.04f));
    points.push_back(Point(-0.597186f, 1.99f, -0.00498f));
    points.push_back(Point(-1.013569f, 0.712263f, 0.262433f));
    points.push_back(Point(0.138381f, 1.99f, -0.612855f));
    points.push_back(Point(1, 0.06261f, -0.427199f));
    points.push_back(Point(-0.592591f, 0.170412f, -0.111682f));
    points.push_back(Point(-0.620757f, 0.001773f, -0.20244f));
    points.push_back(Point(1, 0.97643f, -0.596202f));
    points.push_back(Point(0.875453f, 0.391127f, -1.04f));
    points.push_back(Point(0.799204f, 1.076696f, -1.04f));
    points.push_back(Point(-1.005387f, 0.803385f, -0.70749f));
    points.push_back(Point(-0.640109f, 0.347982f, -1.04f));
    points.push_back(Point(-1.011668f, 1.228813f, -0.720977f));
    points.push_back(Point(-1.006257f, 0.388503f, 0.01565f));
    points.push_back(Point(-0.876297f, 1.99f, -0.805313f));
    points.push_back(Point(-0.480235f, 0.030851f, -1.04f));
    points.push_back(Point(-1.015297f, 1.527297f, -0.809384f));
    points.push_back(Point(1, 1.647515f, -1.002981f));
    points.push_back(Point(-0.999457f, 0.235914f, -0.44111f));
    points.push_back(Point(1, 0.933807f, 0.40358f));
    points.push_back(Point(0.507201f, 0.605645f, -1.04f));
    points.push_back(Point(1, 1.56628f, -0.78155f));
    points.push_back(Point(0.800111f, 0.81069f, -1.04f));
    points.push_back(Point(-1.017031f, 1.399253f, 0.794638f));
    points.push_back(Point(0.587403f, 0.563681f, 0.136418f));
    points.push_back(Point(0.194606f, 1.98f, -0.192581f));
    points.push_back(Point(1, 0.679842f, -0.923592f));
    points.push_back(Point(1, 1.296138f, 0.582966f));
    points.push_back(Point(1, 1.712044f, -0.198413f));
    points.push_back(Point(0.087263f, 1.08538f, -1.04f));
    points.push_back(Point(1, 0.981375f, -0.932605f));
    points.push_back(Point(-0.193753f, 1.139811f, -0.653025f));

    return points;
}

std::vector<Ray> bvh_test_rays_no_inter = {
    Ray(Point(-0.035912f, 0.994493f, -0.065923f), Vector(-0.05945f, -0.126223f, 0.990219f)),
    Ray(Point(-0.035218f, 0.963681f, -0.066142f), Vector(-0.595886f, 0.306108f, 0.74244f)),
    Ray(Point(0.9999f, 1.247962f, 0.754168f), Vector(-0.751472f, -0.311992f, 0.581336f)),
    Ray(Point(0.9999f, 1.778887f, 0.1815f), Vector(-0.690427f, -0.61689f, 0.377832f)),
    Ray(Point(0.9999f, 1.145532f, 0.558433f), Vector(-0.801261f, -0.447637f, 0.396992f)),
    Ray(Point(-0.698062f, 1.9899f, 0.748508f), Vector(0.191626f, -0.949437f, 0.248695f)),
    Ray(Point(-0.022662f, 0.961891f, -0.070107f), Vector(-0.063982f, 0.347521f, 0.935487f)),
    Ray(Point(-0.029636f, 1.033465f, -0.067905f), Vector(-0.479369f, -0.372926f, 0.794438f)),
    Ray(Point(-0.030872f, 0.96567f, -0.067514f), Vector(-0.637002f, 0.310833f, 0.705416f)),
    Ray(Point(-0.031328f, 0.980634f, -0.06737f), Vector(0.464884f, 0.1471f, 0.873066f)),
    Ray(Point(-0.032114f, 0.989257f, -0.067122f), Vector(0.545513f, -0.1871f, 0.816951f)),
    Ray(Point(-0.023089f, 0.97636f, -0.069972f), Vector(0.394092f, -0.125228f, 0.910499f)),
    Ray(Point(-0.023614f, 0.986176f, -0.069806f), Vector(0.681786f, 0.037025f, 0.730614f)),
    Ray(Point(-0.027254f, 0.97859f, -0.068657f), Vector(0.326673f, 0.473001f, 0.818263f)),
    Ray(Point(-0.027939f, 0.994008f, -0.06844f), Vector(0.428401f, 0.191319f, 0.883103f)),
    Ray(Point(-0.019603f, 0.988627f, -0.071073f), Vector(-0.259749f, 0.16663f, 0.951191f)),
    Ray(Point(-0.020404f, 0.994556f, -0.07082f), Vector(0.397208f, -0.178587f, 0.900185f)),
    Ray(Point(-0.015268f, 0.966589f, -0.072442f), Vector(0.21421f, -0.057077f, 0.975119f)),
    Ray(Point(-0.01556f, 0.985813f, -0.072349f), Vector(0.078544f, -0.335131f, 0.938892f)),
    Ray(Point(-0.017948f, 1.036418f, -0.071596f), Vector(0.367794f, 0.221876f, 0.90305f)),
    Ray(Point(-0.006771f, 0.966597f, -0.075125f), Vector(-0.500916f, -0.32322f, 0.802878f)),
    Ray(Point(-0.006857f, 0.971442f, -0.075098f), Vector(-0.46039f, 0.333348f, 0.822752f)),
    Ray(Point(-0.011236f, 0.968456f, -0.073715f), Vector(0.348573f, 0.367558f, 0.862205f)),
    Ray(Point(-0.002683f, 0.969108f, -0.076416f), Vector(0.59674f, -0.339321f, 0.727161f)),
    Ray(Point(-0.003875f, 0.984566f, -0.07604f), Vector(-0.197192f, 0.374394f, 0.90606f)),
    Ray(Point(-0.00403f, 0.988681f, -0.075991f), Vector(0.567583f, 0.540602f, 0.620966f)),
    Ray(Point(-0.00434f, 0.995594f, -0.075893f), Vector(-0.546314f, 0.156433f, 0.822842f)),
    Ray(Point(0.80837f, 1.9899f, 0.277665f), Vector(-0.318681f, -0.680496f, 0.659824f)),
    Ray(Point(-0.016055f, 0.999877f, -0.072193f), Vector(0.512663f, -0.230066f, 0.827192f)),
    Ray(Point(-0.016617f, 1.004109f, -0.072016f), Vector(-0.123507f, -0.551824f, 0.824764f)),
    Ray(Point(-0.016801f, 1.012201f, -0.071958f), Vector(0.204879f, 0.165539f, 0.964687f)),
    Ray(Point(-0.033078f, 1.024464f, -0.066817f), Vector(-0.47875f, 0.135067f, 0.8675f)),
    Ray(Point(-0.008217f, 1.004827f, -0.074669f), Vector(-0.453359f, 0.341986f, 0.823111f)),
    Ray(Point(-0.024591f, 1.000382f, -0.069498f), Vector(-0.409312f, -0.049345f, 0.911059f)),
    Ray(Point(-0.012037f, 1.001371f, -0.073462f), Vector(-0.219374f, -0.099544f, 0.970549f)),
    Ray(Point(-0.012414f, 1.002303f, -0.073343f), Vector(-0.252281f, 0.318064f, 0.913887f)),
    Ray(Point(-0.012619f, 1.013571f, -0.073278f), Vector(-0.200476f, -0.167115f, 0.96534f)),
    Ray(Point(-0.012951f, 1.017294f, -0.073173f), Vector(-0.267893f, -0.366693f, 0.890937f)),
    Ray(Point(-0.013314f, 1.020761f, -0.073059f), Vector(-0.114893f, -0.380838f, 0.917476f)),
    Ray(Point(-0.013147f, 1.025258f, -0.073111f), Vector(-0.554221f, 0.531368f, 0.640692f)),
    Ray(Point(-0.013482f, 1.027952f, -0.073006f), Vector(0.176339f, 0.12769f, 0.976012f)),
    Ray(Point(-0.004176f, 0.999966f, -0.075945f), Vector(-0.105916f, -0.049747f, 0.99313f)),
    Ray(Point(-0.004526f, 1.004082f, -0.075834f), Vector(0.489017f, -0.265653f, 0.830837f)),
    Ray(Point(-0.004705f, 1.010509f, -0.075777f), Vector(0.629759f, -0.381174f, 0.676838f)),
    Ray(Point(-0.024679f, 1.013973f, -0.06947f), Vector(-0.560374f, -0.415334f, 0.716574f)),
    Ray(Point(-0.02905f, 1.01914f, -0.06809f), Vector(0.131341f, -0.452738f, 0.881917f)),
    Ray(Point(-0.029436f, 1.025512f, -0.067968f), Vector(0.344338f, 0.311533f, 0.885651f)),
    Ray(Point(-0.02022f, 1.003031f, -0.070878f), Vector(-0.595479f, 0.117397f, 0.794747f)),
    Ray(Point(0.9999f, 1.501625f, 0.816996f), Vector(-0.94557f, 0.076934f, 0.316194f)),
    Ray(Point(0.576502f, 0.385868f, 0.591703f), Vector(-0.13151f, 0.59357f, 0.793964f)),
    Ray(Point(-1.019796f, 1.969337f, -0.574504f), Vector(0.196876f, -0.561659f, 0.803603f)),
    Ray(Point(0.9999f, 1.536174f, -0.179516f), Vector(-0.217109f, 0.22813f, 0.949116f)),
    Ray(Point(0.9999f, 1.813419f, 0.552326f), Vector(-0.541649f, 0.295292f, 0.787032f)),
    Ray(Point(0.9999f, 1.184152f, -0.296873f), Vector(-0.460832f, -0.21051f, 0.862159f)),
    Ray(Point(-1.019418f, 1.894078f, 0.317029f), Vector(0.237459f, -0.469666f, 0.85031f)),
    Ray(Point(0.226008f, 0.6001f, 0.20114f), Vector(-0.575329f, 0.706275f, 0.41252f)),
    Ray(Point(0.035342f, 1.9899f, 0.671371f), Vector(0.24186f, -0.678137f, 0.693999f)),
    Ray(Point(0.9999f, 1.345194f, -0.047569f), Vector(-0.3267f, -0.712055f, 0.621485f)),
    Ray(Point(0.449549f, 0.6001f, 0.274145f), Vector(0.510451f, 0.122359f, 0.851157f)),
    Ray(Point(-0.620161f, 0.0001f, 0.638983f), Vector(0.411903f, 0.661634f, 0.626559f)),
    Ray(Point(-1.01197f, 0.411882f, 0.783593f), Vector(0.404246f, -0.776516f, 0.483329f)),
    Ray(Point(0.9999f, 1.246021f, 0.628447f), Vector(-0.782965f, -0.313678f, 0.537189f)),
    Ray(Point(0.9999f, 0.518023f, 0.13489f), Vector(-0.092986f, 0.028247f, 0.995267f)),
    Ray(Point(-0.596933f, 1.9899f, 0.670822f), Vector(-0.278219f, -0.853012f, 0.441548f)),
    Ray(Point(-0.506683f, 1.9899f, 0.620413f), Vector(0.26641f, -0.341881f, 0.90119f)),
    Ray(Point(0.9999f, 0.542911f, 0.509043f), Vector(-0.655022f, -0.128964f, 0.744523f)),
    Ray(Point(-0.400249f, 0.0001f, 0.054564f), Vector(-0.3932f, 0.298254f, 0.869735f)),
    Ray(Point(0.889821f, 1.9899f, 0.643035f), Vector(-0.864809f, -0.058775f, 0.498649f)),
    Ray(Point(0.013039f, 0.964779f, -0.081381f), Vector(0.347666f, 0.570614f, 0.743995f)),
    Ray(Point(0.573324f, 0.6001f, 0.172398f), Vector(-0.015748f, 0.177252f, 0.984039f)),
    Ray(Point(0.019224f, 1.9899f, 0.204016f), Vector(-0.506618f, -0.413434f, 0.756578f)),
    Ray(Point(0.061988f, 1.9899f, 0.794787f), Vector(0.176375f, -0.889224f, 0.422105f)),
    Ray(Point(0.017128f, 0.962593f, -0.082672f), Vector(0.270716f, -0.006589f, 0.962637f)),
    Ray(Point(0.025688f, 0.96208f, -0.085375f), Vector(0.3359f, 0.116897f, 0.934616f)),
    Ray(Point(0.416243f, 0.6001f, 0.544809f), Vector(-0.322695f, 0.88636f, 0.332016f)),
    Ray(Point(0.9999f, 1.941455f, 0.600066f), Vector(-0.681872f, -0.549266f, 0.48307f)),
    Ray(Point(0.9999f, 1.121728f, -0.073316f), Vector(-0.765016f, -0.250996f, 0.593087f)),
    Ray(Point(-1.009638f, 0.247244f, 0.585105f), Vector(0.428821f, 0.831056f, 0.3542f)),
    Ray(Point(0.9999f, 0.55367f, 0.541796f), Vector(-0.462738f, -0.687957f, 0.559097f)),
    Ray(Point(-0.865242f, 1.9899f, 0.458778f), Vector(0.717708f, -0.037976f, 0.695308f)),
    Ray(Point(-0.773284f, 0.0001f, 0.587749f), Vector(-0.121091f, 0.914142f, 0.386886f)),
    Ray(Point(-1.01564f, 1.142196f, 0.147094f), Vector(0.763184f, -0.306734f, 0.568739f)),
    Ray(Point(-1.016779f, 1.368842f, 0.575301f), Vector(0.618427f, 0.420462f, 0.663897f)),
    Ray(Point(0.01534f, 1.011077f, -0.082107f), Vector(0.547568f, 0.122819f, 0.827699f)),
    Ray(Point(0.014737f, 1.023388f, -0.081917f), Vector(-0.408717f, -0.079496f, 0.909192f)),
    Ray(Point(0.023751f, 1.00069f, -0.084764f), Vector(-0.541432f, -0.441469f, 0.715512f)),
    Ray(Point(0.121435f, 0.6001f, 0.151576f), Vector(-0.575134f, 0.683914f, 0.448869f)),
    Ray(Point(0.9999f, 0.850421f, 0.145165f), Vector(-0.452241f, 0.53516f, 0.7135f)),
    Ray(Point(-1.01236f, 0.489611f, 0.959613f), Vector(0.427826f, 0.605343f, 0.671212f)),
    Ray(Point(-0.054778f, 1.9899f, 0.956846f), Vector(0.095643f, -0.495364f, 0.863404f)),
    Ray(Point(0.797666f, 1.9899f, 0.403953f), Vector(-0.630618f, -0.727861f, 0.26933f)),
    Ray(Point(-0.773872f, 0.0001f, 0.731986f), Vector(0.648681f, 0.318957f, 0.690999f)),
    Ray(Point(-0.995838f, 0.0001f, 0.555308f), Vector(0.722185f, 0.619237f, 0.308212f)),
    Ray(Point(0.002452f, 1.030921f, -0.078038f), Vector(-0.117464f, -0.441097f, 0.889739f)),
    Ray(Point(0.185121f, 0.6001f, 0.226484f), Vector(-0.150724f, 0.667229f, 0.729443f)),
    Ray(Point(-0.034338f, 0.0001f, 0.229599f), Vector(-0.375925f, 0.283713f, 0.882149f)),
    Ray(Point(0.099134f, 0.6001f, 0.331814f), Vector(0.085276f, 0.636737f, 0.766351f)),
    Ray(Point(-1.017466f, 1.505625f, 0.298463f), Vector(0.108398f, -0.774168f, 0.623629f)),
    Ray(Point(0.9999f, 1.451276f, 0.744477f), Vector(-0.990073f, 0.056077f, 0.128885f)),
    Ray(Point(0.023916f, 1.00475f, -0.084816f), Vector(0.367304f, 0.051793f, 0.928658f)),
    Ray(Point(0.023092f, 1.022576f, -0.084556f), Vector(-0.44043f, -0.018192f, 0.897603f)),
    Ray(Point(0.019494f, 1.013425f, -0.083419f), Vector(0.389264f, 0.263397f, 0.882664f)),
    Ray(Point(0.019029f, 1.023711f, -0.083273f), Vector(0.342799f, 0.626027f, 0.700414f)),
    Ray(Point(0.000613f, 0.971692f, -0.077457f), Vector(-0.508923f, -0.527101f, 0.68056f)),
    Ray(Point(0.000812f, 0.980309f, -0.07752f), Vector(0.262018f, 0.648901f, 0.714335f)),
    Ray(Point(0.000512f, 0.989442f, -0.077425f), Vector(-0.170249f, 0.016199f, 0.985268f)),
    Ray(Point(0.000344f, 0.993622f, -0.077372f), Vector(0.097272f, 0.429747f, 0.897695f)),
    Ray(Point(-0.000126f, 0.993913f, -0.077223f), Vector(-0.438405f, 0.392483f, 0.808553f)),
    Ray(Point(0.008867f, 0.96752f, -0.080063f), Vector(0.544152f, -0.208914f, 0.81256f)),
    Ray(Point(0.009231f, 0.97183f, -0.080178f), Vector(0.161711f, -0.254191f, 0.953539f)),
    Ray(Point(0.008369f, 0.980455f, -0.079906f), Vector(0.324461f, -0.201354f, 0.92422f)),
    Ray(Point(0.016601f, 0.977317f, -0.082505f), Vector(0.042567f, 0.258561f, 0.965057f)),
    Ray(Point(0.01697f, 0.980731f, -0.082622f), Vector(0.119362f, 0.645075f, 0.754739f)),
    Ray(Point(0.008733f, 0.984768f, -0.080021f), Vector(-0.306697f, 0.636372f, 0.707791f)),
    Ray(Point(0.004357f, 0.978386f, -0.078639f), Vector(-0.557076f, 0.438125f, 0.705487f)),
    Ray(Point(0.00392f, 0.992061f, -0.078501f), Vector(-0.288612f, 0.088217f, 0.953373f)),
    Ray(Point(0.006997f, 1.018269f, -0.079473f), Vector(-0.312001f, 0.630199f, 0.710988f)),
    Ray(Point(0.006815f, 1.02612f, -0.079415f), Vector(-0.00784f, -0.260099f, 0.96555f)),
    Ray(Point(0.004079f, 0.995921f, -0.078551f), Vector(-0.498082f, -0.210038f, 0.841308f)),
    Ray(Point(0.012583f, 0.974669f, -0.081237f), Vector(0.086508f, 0.271938f, 0.958419f)),
    Ray(Point(0.01218f, 0.988418f, -0.08111f), Vector(-0.201128f, -0.412113f, 0.888656f)),
    Ray(Point(0.016211f, 0.994879f, -0.082382f), Vector(0.05385f, -0.462084f, 0.8852f)),
    Ray(Point(0.025013f, 0.968179f, -0.085162f), Vector(-0.133447f, -0.47786f, 0.868241f)),
    Ray(Point(0.021507f, 0.96592f, -0.084055f), Vector(-0.244774f, -0.258376f, 0.93452f)),
    Ray(Point(0.029256f, 0.974311f, -0.086502f), Vector(-0.151418f, 0.068475f, 0.986095f)),
    Ray(Point(0.006498f, 1.029662f, -0.079315f), Vector(-0.546127f, 0.142301f, 0.825528f)),
    Ray(Point(0.003877f, 1.004415f, -0.078488f), Vector(0.571606f, 0.511194f, 0.64183f)),
    Ray(Point(0.026728f, 1.037272f, -0.085704f), Vector(0.09746f, -0.349924f, 0.931695f)),
    Ray(Point(0.01883f, 1.031705f, -0.08321f), Vector(-0.204033f, -0.606458f, 0.768491f)),
    Ray(Point(0.9999f, 1.474964f, 0.747578f), Vector(-0.629786f, 0.164854f, 0.759074f)),
    Ray(Point(-1.019144f, 1.83965f, 0.108427f), Vector(0.565618f, 0.110579f, 0.81722f)),
    Ray(Point(-0.748619f, 1.9899f, -0.200296f), Vector(0.422037f, -0.525051f, 0.739058f)),
    Ray(Point(-1.008212f, 1.9899f, 0.425102f), Vector(0.817145f, -0.500016f, 0.286806f)),
    Ray(Point(-0.026607f, 1.9899f, 0.850785f), Vector(0.650773f, -0.024251f, 0.758885f)),
    Ray(Point(-0.563395f, 1.9899f, 0.915257f), Vector(0.513792f, -0.7326f, 0.446447f)),
    Ray(Point(0.052119f, 1.9899f, 0.793421f), Vector(-0.336576f, -0.794999f, 0.504671f)),
    Ray(Point(-1.01036f, 0.45058f, 0.347274f), Vector(0.21141f, 0.341361f, 0.915848f)),
    Ray(Point(0.9999f, 1.280513f, 0.105702f), Vector(-0.700792f, -0.399309f, 0.591137f)),
    Ray(Point(-0.096421f, 0.0001f, 0.05622f), Vector(-0.605183f, 0.022215f, 0.795776f)),
    Ray(Point(-0.966139f, 1.9899f, 0.554426f), Vector(0.101674f, -0.933707f, 0.343299f)),
    Ray(Point(0.601092f, 1.9899f, 0.894116f), Vector(-0.852082f, -0.371196f, 0.369011f)),
    Ray(Point(0.348965f, 1.201217f, -1.0399f), Vector(-0.292857f, -0.207532f, 0.933362f)),
    Ray(Point(0.9999f, 1.74584f, 0.098417f), Vector(-0.568647f, -0.472974f, 0.673005f)),
    Ray(Point(-0.445611f, 0.0001f, 0.146631f), Vector(-0.210686f, 0.648488f, 0.731488f)),
    Ray(Point(-0.151978f, 0.756405f, -0.02927f), Vector(-0.423712f, 0.658491f, 0.621978f)),
    Ray(Point(0.032962f, 0.979326f, -0.087673f), Vector(-0.105916f, -0.049747f, 0.99313f)),
    Ray(Point(0.03244f, 0.992145f, -0.087508f), Vector(0.227066f, -0.036381f, 0.973199f)),
    Ray(Point(0.032069f, 0.996079f, -0.08739f), Vector(-0.220938f, 0.50733f, 0.832948f)),
    Ray(Point(0.037028f, 0.976431f, -0.088957f), Vector(-0.268213f, -0.470982f, 0.84038f)),
    Ray(Point(0.036852f, 0.982151f, -0.088901f), Vector(-0.202903f, -0.448522f, 0.870436f)),
    Ray(Point(0.9999f, 0.87903f, -0.140403f), Vector(-0.82035f, -0.144921f, 0.553194f)),
    Ray(Point(0.31543f, 0.6001f, 0.10868f), Vector(0.192909f, 0.636164f, 0.747049f)),
    Ray(Point(-0.351315f, 1.9899f, 0.155756f), Vector(0.004347f, -0.690173f, 0.723631f)),
    Ray(Point(0.9999f, 0.794155f, 0.497215f), Vector(-0.872273f, -0.061277f, 0.485166f)),
    Ray(Point(0.9999f, 1.343159f, 0.911459f), Vector(-0.110025f, -0.17336f, 0.978693f)),
    Ray(Point(-0.342609f, 0.568736f, 0.030929f), Vector(0.519343f, 0.646506f, 0.55885f)),
    Ray(Point(0.9999f, 1.167192f, 0.152746f), Vector(-0.751911f, -0.295094f, 0.589533f)),
    Ray(Point(-1.014667f, 0.948643f, 0.713242f), Vector(0.502312f, -0.399419f, 0.766907f)),
    Ray(Point(-1.017315f, 1.47553f, 0.836071f), Vector(0.342925f, 0.701978f, 0.624203f)),
    Ray(Point(0.423196f, 1.9899f, 0.285916f), Vector(0.259902f, -0.88975f, 0.375228f)),
    Ray(Point(0.9999f, 0.714179f, 0.080643f), Vector(-0.872515f, 0.126376f, 0.47196f)),
    Ray(Point(0.692004f, 1.9899f, -0.168413f), Vector(0.075291f, -0.641811f, 0.763158f)),
    Ray(Point(0.186633f, 0.6001f, 0.107405f), Vector(0.139389f, 0.245095f, 0.959426f)),
    Ray(Point(0.379683f, 0.6001f, 0.330765f), Vector(-0.085052f, 0.38427f, 0.919295f)),
    Ray(Point(-0.519074f, 1.9899f, 0.093457f), Vector(0.015046f, -0.844603f, 0.535182f)),
    Ray(Point(0.967153f, 1.9899f, 0.508981f), Vector(-0.399141f, -0.751452f, 0.525363f)),
    Ray(Point(0.9999f, 1.299146f, -0.010365f), Vector(-0.830388f, 0.057697f, 0.55419f)),
    Ray(Point(0.9999f, 1.300859f, 0.205213f), Vector(-0.20828f, 0.011781f, 0.977998f)),
    Ray(Point(-0.614127f, 1.9899f, 0.830581f), Vector(0.821011f, -0.506795f, 0.262867f)),
    Ray(Point(-1.015784f, 1.170898f, 0.83952f), Vector(0.681576f, -0.677798f, 0.27576f)),
    Ray(Point(0.490414f, 1.9899f, 0.834645f), Vector(0.042299f, -0.978238f, 0.203128f)),
    Ray(Point(0.9999f, 1.342215f, 0.466224f), Vector(-0.311809f, -0.646851f, 0.69596f)),
    Ray(Point(0.186336f, 1.9899f, 0.717227f), Vector(-0.013858f, -0.062531f, 0.997947f)),
    Ray(Point(-0.566991f, 0.0001f, 0.221087f), Vector(-0.115828f, 0.418815f, 0.900654f)),
    Ray(Point(0.032224f, 0.999795f, -0.087439f), Vector(0.470754f, 0.153715f, 0.86877f)),
    Ray(Point(0.031563f, 1.00999f, -0.08723f), Vector(-0.129557f, -0.485643f, 0.864503f)),
    Ray(Point(0.03571f, 1.007429f, -0.08854f), Vector(-0.626274f, -0.262748f, 0.733992f)),
    Ray(Point(-0.230059f, 0.0001f, 0.159037f), Vector(0.178343f, 0.292142f, 0.939599f)),
    Ray(Point(-1.016648f, 1.342933f, 0.831369f), Vector(0.85456f, 0.049962f, 0.516944f)),
    Ray(Point(-1.01551f, 1.116392f, 0.821339f), Vector(0.944873f, -0.317019f, 0.081945f)),
    Ray(Point(-0.972423f, 1.9899f, 0.646228f), Vector(0.830025f, -0.46328f, 0.310532f)),
    Ray(Point(-0.436774f, 0.0001f, 0.653262f), Vector(0.312942f, 0.540171f, 0.781205f)),
    Ray(Point(-1.012029f, 0.423596f, 0.895762f), Vector(0.595189f, 0.665183f, 0.450868f)),
    Ray(Point(-1.01618f, 1.249667f, 0.03153f), Vector(0.83561f, 0.141994f, 0.530653f)),
    Ray(Point(-0.901564f, 0.0001f, 0.332558f), Vector(0.795261f, 0.496148f, 0.34842f)),
    Ray(Point(-0.259207f, 1.9899f, -0.125108f), Vector(0.50561f, -0.670867f, 0.542491f)),
    Ray(Point(0.528143f, 1.9899f, 0.06674f), Vector(0.265232f, -0.323739f, 0.90821f)),
    Ray(Point(-0.018336f, 1.082918f, -0.27831f), Vector(0.402994f, -0.073484f, 0.912248f)),
    Ray(Point(-1.019475f, 1.905503f, 0.942847f), Vector(0.968814f, 0.180452f, 0.169811f)),
    Ray(Point(0.469164f, 0.6001f, 0.674555f), Vector(0.046172f, 0.026928f, 0.998571f)),
    Ray(Point(0.9999f, 1.064214f, 0.773441f), Vector(-0.302197f, 0.12337f, 0.945228f)),
    Ray(Point(0.179529f, 0.6001f, 0.434863f), Vector(-0.232525f, 0.602973f, 0.763122f)),
    Ray(Point(0.9999f, 1.104419f, 0.684943f), Vector(-0.347368f, -0.756884f, 0.553589f)),
    Ray(Point(-0.481329f, 1.139742f, -1.0399f), Vector(0.530766f, 0.20969f, 0.821168f)),
    Ray(Point(0.532205f, 0.11818f, 0.742832f), Vector(0.581557f, 0.730607f, 0.357779f)),
    Ray(Point(0.9999f, 1.712044f, -0.198413f), Vector(-0.099093f, -0.611609f, 0.78493f)),
    Ray(Point(-0.149006f, 0.0001f, 0.785647f), Vector(-0.473158f, 0.679404f, 0.560831f)),
    Ray(Point(0.924601f, 1.9899f, 0.387681f), Vector(-0.4252f, -0.786168f, 0.448491f)),
    Ray(Point(-0.425263f, 0.112378f, 0.05703f), Vector(0.243763f, 0.570154f, 0.78454f)),
    Ray(Point(-0.953761f, 0.0001f, 0.748978f), Vector(0.905763f, 0.008025f, 0.42371f)),
    Ray(Point(-0.316937f, 0.403011f, 0.022822f), Vector(-0.282526f, 0.736802f, 0.614249f)),
    Ray(Point(-1.011165f, 0.508518f, 0.340283f), Vector(0.537372f, -0.117262f, 0.835153f)),
    Ray(Point(-1.016291f, 1.271894f, 0.368953f), Vector(0.758673f, 0.342987f, 0.553873f)),
    Ray(Point(-0.121812f, 1.9899f, 0.598193f), Vector(-0.32693f, -0.814816f, 0.47874f)),
    Ray(Point(-1.013913f, 0.798578f, 0.672761f), Vector(0.61893f, 0.085261f, 0.780805f)),
    Ray(Point(0.9999f, 1.159845f, 0.098417f), Vector(-0.651096f, -0.128603f, 0.748021f)),
    Ray(Point(0.9999f, 0.98822f, 0.585551f), Vector(-0.640136f, 0.642766f, 0.420807f)),
    Ray(Point(-1.016931f, 1.399253f, 0.794638f), Vector(0.664408f, 0.59159f, 0.45671f)),
    Ray(Point(0.194606f, 1.9799f, -0.192581f), Vector(0.000057f, -0.351953f, 0.936018f)),
    Ray(Point(0.9999f, 0.981375f, -0.932605f), Vector(-0.101229f, -0.119253f, 0.98769f)),
    Ray(Point(-0.195616f, 0.0001f, 0.155778f), Vector(-0.142098f, 0.327295f, 0.934177f)),
    Ray(Point(-1.014279f, 0.87139f, 0.752203f), Vector(0.181673f, 0.564353f, 0.805295f)),
    Ray(Point(-0.203015f, 0.288908f, -0.013154f), Vector(-0.076526f, 0.384956f, 0.919757f)),
    Ray(Point(-1.019372f, 1.884963f, 0.944245f), Vector(0.383668f, -0.53128f, 0.755341f)),
    Ray(Point(-1.01341f, 0.6985f, 0.662047f), Vector(0.278323f, 0.787591f, 0.549761f)),
    Ray(Point(0.958453f, 0.0001f, 0.856725f), Vector(0.012327f, 0.885402f, 0.464662f)),
    Ray(Point(0.9999f, 1.148259f, 0.818478f), Vector(-0.870957f, -0.39589f, 0.291042f)),
    Ray(Point(0.9999f, 0.473335f, 0.288567f), Vector(-0.220256f, 0.85865f, 0.462826f)),
    Ray(Point(-0.207724f, 1.9899f, -0.828863f), Vector(0.500069f, -0.009154f, 0.865937f)),
    Ray(Point(0.9999f, 1.216356f, 0.082897f), Vector(-0.796616f, -0.235743f, 0.556622f)),
    Ray(Point(-0.999555f, 0.0001f, 0.838598f), Vector(0.349219f, 0.54842f, 0.75979f)),
    Ray(Point(0.004728f, 0.0001f, 0.762281f), Vector(-0.713338f, 0.037451f, 0.699819f))
};

std::vector<Ray> bvh_test_rays_inter = create_ray_inter_vector();
std::vector<Point> bvh_test_rays_inter_result_points = create_ray_inter_result_points();

#endif
