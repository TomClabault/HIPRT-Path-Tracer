/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
#version 430

// 0.0f gives 100% of texture 1
// 1.0f gives 100% of texture_2
uniform float u_blend_factor;

uniform sampler2D u_texture_1;
uniform sampler2D u_texture_2;

// How many samples to scale texture 1 and 2 by
uniform int u_sample_number_1;
uniform int u_sample_number_2;

uniform int u_resolution_scaling;

uniform float u_gamma;
uniform float u_exposure;
uniform int u_do_tonemapping;

#ifdef COMPUTE_SCREENSHOTER
uniform layout(binding = 2, rgba8ui) writeonly uimage2D u_output_image;
#else
in vec2 vs_tex_coords;
out vec4 out_color;
#endif // COMPUTE_SCREENSHOTER

#ifdef COMPUTE_SCREENSHOTER
layout(local_size_x = 8, local_size_y = 8) in;
#endif // COMPUTE_SCREENSHOTER

void main()
{
#ifdef COMPUTE_SCREENSHOTER																		
	ivec2 dims = textureSize(u_texture_1, 0);													
	ivec2 thread_id = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);				
	if (thread_id.x >= dims.x || thread_id.y >= dims.y)							
		return;

	vec4 hdr_color_1 = texelFetch(u_texture_1, thread_id / u_resolution_scaling, 0);
	vec4 hdr_color_2 = texelFetch(u_texture_2, thread_id / u_resolution_scaling, 0);
#else
	vec4 hdr_color_1 = texture(u_texture_1, vs_tex_coords / u_resolution_scaling);
	vec4 hdr_color_2 = texture(u_texture_2, vs_tex_coords / u_resolution_scaling);
#endif

	vec4 final_color_1 = hdr_color_1;
	vec4 final_color_2 = hdr_color_2;

	// Scaling by sample count
	final_color_1 = final_color_1 / float(max(1, u_sample_number_1));
	final_color_2 = final_color_2 / float(max(1, u_sample_number_2));
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped_1 = 1.0f - exp(-final_color_1 * u_exposure);
		vec4 tone_mapped_2 = 1.0f - exp(-final_color_2 * u_exposure);

		final_color_1 = pow(tone_mapped_1, vec4(1.0f / u_gamma));
		final_color_2 = pow(tone_mapped_2, vec4(1.0f / u_gamma));
	}

	final_color_1 = vec4(final_color_1.rgb, 1.0f);
	final_color_2 = vec4(final_color_2.rgb, 1.0f);

	vec4 blended_color = final_color_1 * (1.0f - u_blend_factor) + final_color_2 * u_blend_factor;
#ifdef COMPUTE_SCREENSHOTER
	uvec4 ublended_color = uvec4(blended_color * 255);
	imageStore(u_output_image, thread_id, ublended_color);
#else
	out_color = blended_color;
#endif // COMPUTE_SCREENSHOTER
};
