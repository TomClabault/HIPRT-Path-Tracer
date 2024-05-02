/*
 * Copyright 2024 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */
 
 #version 330

out vec2 vs_tex_coords;

void main()
{
	vec2 triangle_vertices[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));
	vec2 triangle_tex_coords[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(0, 1), vec2(1, 0), vec2(1, 1), vec2(0, 1));

	gl_Position = vec4(triangle_vertices[gl_VertexID], 1, 1);
	vs_tex_coords = triangle_tex_coords[gl_VertexID];
};
