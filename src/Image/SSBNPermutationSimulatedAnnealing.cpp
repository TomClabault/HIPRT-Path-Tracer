/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#include "Device/includes/SSBNPermutation/SSBNPermutationCommon.h"
#include "HostDeviceCommon/Xorshift.h"
#include "Image/SSBNPermutationSimulatedAnnealing.h"
#include "Utils/Utils.h"

#include <random>

SSBNPermutationSimulatedAnnealing::SSBNPermutationSimulatedAnnealing(const Image8Bit& blue_noise_input_image, int max_allowed_permutation_distance)
{
	unsigned int input_width  = blue_noise_input_image.width;
	unsigned int input_height = blue_noise_input_image.height;

	m_blue_noise_image_original = blue_noise_input_image;
	m_blue_noise_image			= Image8Bit(input_width, input_height, 1);
	// Copying only the first channel of the input image to m_blue_noise_image
	for (unsigned int y = 0; y < input_height; y++)
		for (unsigned int x = 0; x < input_width; x++)
			m_blue_noise_image.data()[x + y * input_width] = blue_noise_input_image.data()[(x + y * input_width) * blue_noise_input_image.channels + 0];

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

	m_max_allowed_permutation_distance = max_allowed_permutation_distance;
}

void SSBNPermutationSimulatedAnnealing::compute_permutation()
{
	unsigned int input_width  = m_blue_noise_image.width;
	unsigned int input_height = m_blue_noise_image.height;

	m_permuted_positions = std::vector<int>(input_width * input_height);
	for (int i = 0; i < input_width * input_height; i++)
		m_permuted_positions[i] = i;

	double temperature		 = 1;
	double cooling_rate		 = 0.99999999;
	double current_mse_error = 10000.0;

	std::mt19937 rng(1337);
	std::uniform_int_distribution<int> dist_x(0, input_width - 1);
	std::uniform_int_distribution<int> dist_y(0, input_height - 1);
	std::uniform_int_distribution<int> dist_radius(-m_max_allowed_permutation_distance, m_max_allowed_permutation_distance);
	Xorshift32Generator dist_prob(42);

	// Running the random permutation loop
	//
	// Enough iterations to basically give each pixel a chance to be swapped with a good candidate in its neighborhood
	size_t iter		  = 0;
	double target_mse = 5.0;
	while (current_mse_error > target_mse)
	{
		if (iter++ % 100000000 == 0)
		{
			Image8Bit current_t_1_result(input_width, input_height, 1);
			for (unsigned int index = 0; index < input_width * input_height; index++)
				current_t_1_result.data()[index] = m_blue_noise_image_original.data()[m_permuted_positions[index] * m_blue_noise_image_original.channels];

			current_mse_error = 0.0;
			for (unsigned int index = 0; index < input_width * input_height; index++)
				current_mse_error += hippt::square(current_t_1_result.data()[index] - m_blue_noise_image_t_plus_1.data()[index]);
			current_mse_error /= (input_width * input_height);

			std::cout << "Iteration " << iter << ", Temperature: " << temperature << ". Current state MSE: " << current_mse_error
					  << ". Target MSE: " << target_mse << std::endl;
		}

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
		// If the total permutation distance of the chosen pixel is greater than the radius, we're not allowing the permutation. We want to produce local
		// permutations from t_0 to t_+1 so we only allow small permutations
		{
			int new_permutation_index = m_permuted_positions[indexB];
			int new_permutation_x	  = new_permutation_index % input_width;
			int new_permutation_y	  = new_permutation_index / input_width;

			float distance_after_swap = hippt::length(make_float2(new_permutation_x - random_x, new_permutation_y - random_y));

			if (distance_after_swap > m_max_allowed_permutation_distance)
				continue;
		}

		// Decide whether to accept the swap
		bool accept = false;
		if (delta_error <= 1.0e-6f)
			// It's an improvement, always accept
			accept = true;
		else
		{
			// It's worse, accept with a probability based on temperature
			float acceptance_probability = std::exp(-delta_error / temperature);
			if (dist_prob() < acceptance_probability)
				accept = true;
		}

		// Apply the swap if accepted
		if (accept)
		{
			// Swap the values in our working state
			std::swap(m_blue_noise_image[indexA], m_blue_noise_image[indexB]);

			std::swap(m_permuted_positions[indexA], m_permuted_positions[indexB]);
		}

		// Cool down
		temperature *= cooling_rate;

		if (current_mse_error <= 2.5)
			// Good enough
			break;
	}

	Image8Bit final_permutation_map_visualization(input_width, input_height, 1);
	for (unsigned int index = 0; index < input_width * input_height; index++)
		final_permutation_map_visualization.data()[index] =
								m_blue_noise_image_original.data()[m_permuted_positions[index] * m_blue_noise_image_original.channels];

	double final_mse = 0.0;
	for (unsigned int index = 0; index < input_width * input_height; index++)
		final_mse += hippt::square(final_permutation_map_visualization.data()[index] - m_blue_noise_image_t_plus_1.data()[index]);
	final_mse /= (input_width * input_height);
	std::cout << "Final MSE error: " << final_mse << std::endl;
}

void SSBNPermutationSimulatedAnnealing::write_permutations_to_file(const std::string_view file_path)
{
	std::ofstream permutation_output_file(file_path.data(), std::ios::binary);
	permutation_output_file.write(reinterpret_cast<const char*>(m_permuted_positions.data()), m_permuted_positions.size() * sizeof(int));
}

void SSBNPermutationSimulatedAnnealing::write_permutation_visualization_image(const std::string_view file_path)
{
	unsigned int input_width  = m_blue_noise_image.width;
	unsigned int input_height = m_blue_noise_image.height;

	Image8Bit final_permutation_map_visualization(input_width, input_height, 1);
	for (unsigned int index = 0; index < input_width * input_height; index++)
		final_permutation_map_visualization.data()[index] =
								m_blue_noise_image_original.data()[m_permuted_positions[index] * m_blue_noise_image_original.channels];

	final_permutation_map_visualization.write_image_png(file_path);
}

std::vector<int>& SSBNPermutationSimulatedAnnealing::permuted_positions()
{
	return m_permuted_positions;
}
