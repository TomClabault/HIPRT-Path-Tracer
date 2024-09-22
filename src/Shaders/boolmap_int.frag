/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
 #version 430

// This is a 'scalar' texture, containing data only in the red channel
// In this shader, it represents the sample count per pixel
uniform isampler2D u_texture;
uniform int u_resolution_scaling;
uniform float u_threshold_val;

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

	// We're using abs() here because the sampling count can be negative if 
	// the pixel isn't being sampled anymore (it has converged and has been 
	// excluded by the adaptive sampling)
	float scalar = texelFetch(u_texture, thread_id / u_resolution_scaling, 0).r;
#else
	float scalar = texture(u_texture, vs_tex_coords / u_resolution_scaling).r;
#endif
	
	vec4 final_color = vec4(0.0f);
	if (scalar < u_threshold_val && scalar != -1.0f)
		// If the value of the scalar is -1, this is a special value which is used
		// by the adaptive sampling to indicate that a pixel has not converged yet.
		// If the pixel has not converged yet, then it must have the "hotter" color
		// which the color of the u_max_val
		//
		// We only set the pixel white if it has a value lower than the threshold
		// number. The threshold is the current sample count so if the pixel has a
		// sample count lower than that, that means that it has converged
		final_color = vec4(1.0f);

#ifdef COMPUTE_SCREENSHOTER
	uvec4 ufinal_color = uvec4(final_color * 255.0f);
	imageStore(u_output_image, thread_id, ufinal_color);
#else
	out_color = final_color;
#endif // COMPUTE_SCREENSHOTER
};
