/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/SSBNPermutation/SSBNPermutationCommon.h"
#include "Image/SSBNPermutationSimulatedAnnealing.h"

#include <random>

SSBNPermutationSimulatedAnnealing::SSBNPermutationSimulatedAnnealing(const Image8Bit& blue_noise_image)
{
	unsigned int input_width  = blue_noise_image.width;
	unsigned int input_height = blue_noise_image.height;

	m_blue_noise_image = Image8Bit(input_width, input_height, 1);
	// Copying only the first channel of the input image to m_blue_noise_image
	for (unsigned int y = 0; y < input_height; y++)
		for (unsigned int x = 0; x < input_width; x++)
			m_blue_noise_image.data()[x + y * input_width] = blue_noise_image.data()[(x + y * input_width) * blue_noise_image.channels + 0];

	m_blue_noise_image_t_plus_1 = Image8Bit(input_width, input_height, 1);

	int blue_noise_offset_x, blue_noise_offset_y;
	get_blue_noise_texture_offset(input_width, input_height, 0, blue_noise_offset_x, blue_noise_offset_y);

	for (int y = 0; y < input_height; y++)
	{
		for (int x = 0; x < input_width; x++)
		{
			int blue_noise_index_x = (x + blue_noise_offset_x) % input_width;
			int blue_noise_index_y = (y + blue_noise_offset_y) % input_height;

			m_blue_noise_image_t_plus_1.data()[x + y * input_width] = m_blue_noise_image.data()[blue_noise_index_x + blue_noise_index_y * input_width];
		}
	}

	m_permuted_positions = std::vector<int>(input_width * input_height);
	for (int i = 0; i < input_width * input_height; i++)
		m_permuted_positions[i] = i;

	double temperature	= 1000.0;
	double cooling_rate = 0.99999999;
	int max_radius		= 6;

	std::mt19937 rng(1337);
	std::uniform_int_distribution<int> dist_x(0, input_width - 1);
	std::uniform_int_distribution<int> dist_y(0, input_height - 1);
	std::uniform_int_distribution<int> dist_radius(-max_radius, max_radius);
	std::uniform_real_distribution<float> dist_prob(0.0f, 1.0f);

	// Running the random permutation loop
	//
	// Enough iterations to basically give each pixel a chance to be swapped with a good candidate in its neighborhood
	size_t max_iter = input_width * input_height * max_radius * 6 * 1000;
	for (size_t iter = 0; iter < max_iter; ++iter)
	{
		// Randomly select a pixel and a small random offset
		int random_x		= dist_x(rng);
		int random_y		= dist_y(rng);
		int random_offset_x = dist_radius(rng);
		int random_offset_y = dist_radius(rng);
		int new_x			= (random_x + random_offset_x + input_width) % input_width;
		int new_y			= (random_y + random_offset_y + input_height) % input_height;

		int indexA = random_x + random_y * input_width;
		int indexB = new_x + new_y * input_width;

		if (random_x == new_x && random_y == new_y)
			// Skip if the same pixel is selected
			continue;

		float current_error = hippt::abs(m_blue_noise_image.data()[indexA] - m_blue_noise_image_t_plus_1.data()[indexA]) +
							  hippt::abs(m_blue_noise_image.data()[indexB] - m_blue_noise_image_t_plus_1.data()[indexB]);

		float swapped_error = hippt::abs(m_blue_noise_image.data()[indexB] - m_blue_noise_image_t_plus_1.data()[indexA]) +
							  hippt::abs(m_blue_noise_image.data()[indexA] - m_blue_noise_image_t_plus_1.data()[indexB]);

		float delta_error = swapped_error - current_error;

		// Decide whether to accept the swap
		bool accept = false;
		if (delta_error < 0.0f)
			// It's an improvement, always accept
			accept = true;
		else
		{
			// It's worse, accept with a probability based on temperature
			float acceptance_probability = std::exp(-delta_error / temperature);
			if (dist_prob(rng) < acceptance_probability)
				accept = true;
		}

		// Apply the swap if accepted
		if (accept)
		{
			// Swap the visual values in our working state
			std::swap(m_blue_noise_image[indexA], m_blue_noise_image[indexB]);

			std::swap(m_permuted_positions[indexA], m_permuted_positions[indexB]);
		}

		// Cool down
		temperature *= cooling_rate;

		if (iter % 5000000 == 0)
			std::cout << "Iteration " << iter << " (" << iter / (double)max_iter * 100.0 << "%), Temperature: " << temperature << std::endl;
	}

	std::ofstream permutation_output_file("permutation.bin", std::ios::binary);
	permutation_output_file.write(reinterpret_cast<const char*>(m_permuted_positions.data()), m_permuted_positions.size() * sizeof(int));

	Image8Bit final_permutation_map_visualization(input_width, input_height, 1);
	for (unsigned int index = 0; index < input_width * input_height; index++)
		final_permutation_map_visualization.data()[index] = blue_noise_image.data()[m_permuted_positions[index] * blue_noise_image.channels];

	double final_mse = 0.0;
	for (unsigned int index = 0; index < input_width * input_height; index++)
		final_mse += hippt::square(final_permutation_map_visualization.data()[index] - m_blue_noise_image_t_plus_1.data()[index]);
	final_mse /= (input_width * input_height);
	std::cout << "Final MSE error: " << final_mse << std::endl;
}
