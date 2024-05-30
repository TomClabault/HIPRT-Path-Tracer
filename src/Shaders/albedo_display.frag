/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
 #version 430

uniform sampler2D u_texture;
uniform int u_sample_number;
uniform int u_resolution_scaling;

#ifdef COMPUTE_SCREENSHOTER
layout(binding = 2, rgba8) writeonly uniform image2D u_output_image;
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

	imageStore(u_output_image, thread_id, texelFetch(u_texture, thread_id / u_resolution_scaling, 0));
#else
	out_color = texture(u_texture, vs_tex_coords / u_resolution_scaling);
#endif
};