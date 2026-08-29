/*
 * Copyright 2026 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMAGE_SSBN_PERMUTATION_SIMULATED_ANNEALING_H
#define IMAGE_SSBN_PERMUTATION_SIMULATED_ANNEALING_H

#include "Image/Image.h"

class SSBNPermutationSimulatedAnnealing
{
public:
	SSBNPermutationSimulatedAnnealing(const Image8Bit& blue_noise_input_image, int max_allowed_permutation_distance, int max_time_seconds = 600);

	void compute_permutation();
	void write_permutations_to_file(const std::string_view file_path);
	void write_permutation_visualization_image(const std::string_view file_path);

	std::vector<int>& permuted_positions();

private:
	Image8Bit m_blue_noise_image_original;

	Image8Bit m_blue_noise_image;
	Image8Bit m_blue_noise_image_t_plus_1;

	int m_max_allowed_permutation_distance;
	int m_max_time_seconds;

	std::vector<int> m_permuted_positions;
};

#endif // #ifndef IMAGE_SSBN_PERMUTATION_SIMULATED_ANNEALING_H
