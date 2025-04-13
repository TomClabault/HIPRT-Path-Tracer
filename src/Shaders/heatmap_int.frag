/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
 #version 430

// This is a 'scalar' texture, containing data only in the red channel
// In this shader, it represents the sample count per pixel
uniform isampler2D u_texture;
uniform int u_resolution_scaling;

// This shader supports up to 16 color stops. This doesn't mean that
// the user has to provide 16 stops. The user only provides X stops as
// indicated by u_nb_stops
uniform vec3 u_color_stops[16];
uniform int u_nb_stops;

uniform float u_min_val;
uniform float u_max_val;

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
	
	if (u_min_val == u_max_val)
	{
		// If the bounds are the same, arbitrarily choosing
		// to color with the last color stop
#ifdef COMPUTE_SCREENSHOTER
	uvec4 output_color = uvec4(uvec3(u_color_stops[u_nb_stops - 1] * 255), 255);
	imageStore(u_output_image, thread_id, output_color);
#else
	out_color = vec4(u_color_stops[u_nb_stops - 1], 1.0f);
#endif

		return;
	}

	if (scalar == -1.0f)
		// If the value of the scalar is -1, this is a special value which is used
		// by the adaptive sampling to indicate that a pixel has not converged yet.
		// If the pixel has not converged yet, then it must have the "hotter" color
		// which the color of the u_max_val
		scalar = u_max_val;

	scalar = clamp(scalar, u_min_val, u_max_val);
	// Brings scalar between 0 and 1 relative to u_min_val and u_max_val
	float normalized = (scalar - u_min_val) / (u_max_val - u_min_val);

	// This indicates the stop to use but this is a float so it could be 1.5 for example
	// which would mean that we would have to pick 50% of u_color_stops[1] + 50% of u_color_stops[2]
	float stop = normalized * (u_nb_stops - 1);

	int low_stop = int(floor(stop));
	int high_stop = int(ceil(stop));
	float fraction_of_high_stop = stop - low_stop;

	// Lerping between the 2 stops
	// Example:
	// For a 'stop' value of 2.37, we get low_stop = 2, high_stop = 3
	// and fraction_of_high_stop = 0.37
	// We're closer to stop 2 than stop 3 so we should interpolate more of stop 2 than stop 3
	// This gives us 
	//
	// out_color = stop2 * (1.0f - 0.37) + stop3 * 0.37 
	//
	// which is more of stop2 than stop3
	vec4 final_color = vec4(mix(u_color_stops[low_stop], u_color_stops[high_stop], fraction_of_high_stop), 1.0f);
#ifdef COMPUTE_SCREENSHOTER
	uvec4 ufinal_color = uvec4(final_color * 255.0f);
	imageStore(u_output_image, thread_id, ufinal_color);
#else
	out_color = final_color;
#endif // COMPUTE_SCREENSHOTER
};
