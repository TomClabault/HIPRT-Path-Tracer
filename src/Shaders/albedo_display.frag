#version 430

uniform sampler2D u_texture;
uniform int u_sample_number;

in vec2 vs_tex_coords;
out vec4 out_color;

void main()
{
	out_color = texture(u_texture, vs_tex_coords);
};