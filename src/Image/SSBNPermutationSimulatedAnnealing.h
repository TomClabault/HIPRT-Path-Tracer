/*
 * Copyright 2025 Tom Clabault. GNU GPL3 license.
 * GNU GPL3 license copy: https://www.gnu.org/licenses/gpl-3.0.txt
 */

#ifndef IMAGE_SSBN_PERMUTATION_SIMULATED_ANNEALING_H
#define IMAGE_SSBN_PERMUTATION_SIMULATED_ANNEALING_H

#include "Image/Image.h"

class SSBNPermutationSimulatedAnnealing
{
public:
	SSBNPermutationSimulatedAnnealing(const Image8Bit& blue_noise_image);

	void compute_permutation(Image32Bit& out_permutation) const;

private:
	Image8Bit m_blue_noise_image;
	Image8Bit m_blue_noise_image_t_plus_1;

	std::vector<int> m_permuted_positions;
};

#endif
