/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
 #version 430

uniform sampler2D u_texture;
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
	ivec2 dims = textureSize(u_texture, 0);
	ivec2 thread_id = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);				
	if (thread_id.x >= dims.x || thread_id.y >= dims.y)							
		return;

	vec4 hdr_color = texelFetch(u_texture, thread_id / u_resolution_scaling, 0);
#else
	vec4 hdr_color = texture(u_texture, vs_tex_coords / u_resolution_scaling);
#endif

	vec4 final_color = hdr_color;
	// Remapping normals for displaying
	final_color = (final_color + 1.0f) * 0.5f;
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped = 1.0f - exp(-final_color * u_exposure);
		final_color = pow(tone_mapped, vec4(1.0f / u_gamma));
	}

	final_color = vec4(final_color.rgb, 1.0f);

#ifdef COMPUTE_SCREENSHOTER
	uvec4 ufinal_color = uvec4(final_color * 255);
	imageStore(u_output_image, thread_id, ufinal_color);
#else
	out_color = final_color;
#endif
};