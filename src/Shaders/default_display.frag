/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
#version 430

uniform sampler2D u_texture;
uniform int u_sample_number;

uniform float u_gamma;
uniform float u_exposure;
uniform int u_do_tonemapping;

#ifdef COMPUTE_SCREENSHOTER
layout(binding = 2, rgba8) writeonly uniform image2D u_output_image;
#else
in vec2 vs_tex_coords;
out vec4 out_color;
#endif // COMPUTE_SCREENSHOTER

#ifdef COMPUTE_SCREENSHOTER
layout(local_size_x = 8, local_size_y = 8) in;
#endif // COMPUTE_SCREENSHOTER

float Luminance(vec4 color)
{
	return 0.212671f * color.r + 0.715160f * color.g + 0.072169f * color.b;
}

void main()
{
#ifdef COMPUTE_SCREENSHOTER																		
	ivec2 dims = textureSize(u_texture, 0);													
	ivec2 thread_id = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);				
	if (thread_id.x >= dims.x || thread_id.y >= dims.y)							
		return;

	vec4 hdr_color = texelFetch(u_texture, thread_id, 0);
#else
	vec4 hdr_color = texture(u_texture, vs_tex_coords);
#endif

	vec4 final_color = hdr_color;
	// Scaling by sample count
	final_color = final_color / float(u_sample_number);
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped = 1.0f - exp(-final_color * u_exposure);
		final_color = pow(tone_mapped, vec4(1.0f / u_gamma));
	}

	final_color = vec4(final_color.rgb, 1.0f);

#ifdef COMPUTE_SCREENSHOTER
	imageStore(u_output_image, thread_id, final_color);
#else
	out_color = final_color;
#endif // COMPUTE_SCREENSHOTER
};