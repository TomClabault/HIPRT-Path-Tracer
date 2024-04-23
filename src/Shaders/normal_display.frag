#version 430

uniform sampler2D u_texture;
uniform int u_sample_number;

uniform float u_gamma;
uniform float u_exposure;
uniform int u_do_tonemapping;

in vec2 vs_tex_coords;
out vec4 out_color;

void main()
{
	vec4 hdr_color = texture(u_texture, vs_tex_coords);

	vec4 final_color = hdr_color;
	// Scaling by sample count
	final_color = final_color / float(u_sample_number);
	// Remapping normals for displaying
	final_color = (final_color + 1.0f) * 0.5f;
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped = 1.0f - exp(-final_color * u_exposure);
		final_color = pow(tone_mapped, vec4(1.0f / u_gamma));
	}

	final_color = vec4(final_color.rgb, 1.0f);

	out_color = final_color;
};