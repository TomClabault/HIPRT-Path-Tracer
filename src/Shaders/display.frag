#version 330

uniform sampler2D u_texture;
uniform int u_sample_number;
uniform float u_gamma;
uniform float u_exposure;
uniform int u_do_tonemapping;
uniform int u_display_normals;
uniform int u_scale_by_frame_number;
uniform int u_sample_count_override;

in vec2 vs_tex_coords;

void main()
{
	vec4 hdr_color = texture(u_texture, vs_tex_coords);
	if (u_scale_by_frame_number == 1)
		hdr_color = hdr_color / float(u_sample_count_override != -1 ? (u_sample_count_override) : (u_sample_number));
		
	if (u_display_normals == 1)
		hdr_color = (hdr_color + 1.0f) * 0.5f; // Remapping normals
		
	if (u_do_tonemapping == 1)
	{
		vec4 tone_mapped = 1.0f - exp(-hdr_color * u_exposure);
		vec4 gamma_corrected = pow(tone_mapped, vec4(1.0f / u_gamma));
		gl_FragColor = vec4(gamma_corrected.rgb, 1.0f);
	}
	else
	{
		gl_FragColor = vec4(hdr_color.rgb, 1.0f);
	}
};
