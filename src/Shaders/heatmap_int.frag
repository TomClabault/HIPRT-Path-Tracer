#version 430

// This is a 'scalar' texture, containing data only in the red channel
uniform sampler2D u_texture;

// This shader supports up to 16 color stops. This doesn't mean that
// the user has to provide 16 stops. The user only provides X stops as
// indicated by u_nb_stops
uniform vec4 u_color_stops[16];
uniform int u_nb_stops;

uniform float u_min_val;
uniform float u_max_val;

in vec2 vs_tex_coords;

out vec4 out_color;

void main()
{
	float scalar = texture(u_texture, vs_tex_coords).r;

	// Brings scalar between 0 and 1 relative to u_min_val and u_max_val
	float normalized = (scalar - u_min_val) / (u_max_val - u_min_val);

	// This indicates the stop to use but this is a float so it could be 1.5 for example
	// which would mean that we would have to pick 50% of u_color_stops[1] + 50% of u_color_stops[2]
	float stop = normalized * u_nb_stops;

	int low_stop = int(floor(stop));
	int high_stop = int(ceil(stop));
	float fraction = stop - low_stop;

	// Lerping between the 2 stops
	// Example:
	// For a 'stop' value of 2.37, we get low_stop = 2, high_stop = 3
	// and fraction = 0.37
	// We're closer to stop 2 than stop 3 so we should interpolate more of stop 2 than stop 3
	// This gives us 
	//
	// out_color = stop2 * (1.0f - 0.37) + stop3 * 0.37 
	//
	// which is more of stop2 than stop3
	out_color = u_color_stops[low_stop] * (1.0f - fraction) + u_color_stops[high_stop] * fraction;
};