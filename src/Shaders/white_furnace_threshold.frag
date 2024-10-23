/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 

/**
 * This shader is meant to be used with a white furnace scene and
 * it will highlight in green and red pixel values that are below
 * or above 0.5f respectively. 
 * 0.5f is the correct assumed value for a furnace test on a material
 * with 1.0f albedo and 0.5f ambient lighting
 */
#version 430

uniform sampler2D u_texture;
uniform int u_sample_number;
uniform int u_resolution_scaling;

uniform float u_gamma;
uniform float u_exposure;
uniform int u_do_tonemapping;

// If true, the shader will display
// pixel that lose energy as green. Pixels will not be highlighted
// if false
uniform bool u_use_low_threshold;
// If true, the shader will display
// pixel that gain energy as red. Pixels will not be highlighted
// if false
uniform bool u_use_high_threshold;

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
	ivec2 dims = textureSize(u_texture, 0);													
	ivec2 thread_id = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);				
	if (thread_id.x >= dims.x || thread_id.y >= dims.y)							
		return;

	vec4 hdr_color = texelFetch(u_texture, thread_id / u_resolution_scaling, 0);
#else
	vec4 hdr_color = texture(u_texture, vs_tex_coords / u_resolution_scaling);
#endif

	vec4 final_color = hdr_color;
	// Scaling by sample count
	final_color = final_color / float(u_sample_number);
	
	if ((final_color.r > 0.51f || final_color.g > 0.51f || final_color.b > 0.51f) && u_use_high_threshold)
		final_color = vec4(1.0f, 0.0f, 0.0f, 1.0f);
	if ((final_color.r < 0.49f || final_color.g < 0.49f || final_color.b < 0.49f) && u_use_low_threshold)
		final_color = vec4(0.0f, 1.0f, 0.0f, 1.0f);
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped = 1.0f - exp(-final_color * u_exposure);
		final_color = pow(tone_mapped, vec4(1.0f / u_gamma));
	}

	final_color = vec4(final_color.rgb, 1.0f);

#ifdef COMPUTE_SCREENSHOTER
	imageStore(u_output_image, thread_id, uvec4(final_color * 255));
#else
	out_color = final_color;
#endif // COMPUTE_SCREENSHOTER
};
