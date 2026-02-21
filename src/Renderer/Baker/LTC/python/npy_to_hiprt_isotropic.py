#!/usr/bin/env python3
"""
Generate a ready-to-use C++ LTC parameter table directly from the .npy inputs.
This replaces the two-step pipeline: it loads the ltc matrix and amp .npy files,
performs the same normalization and row/column math as your post-processing
script, and writes the final annotated C++ array using make_float4(...) entries.

Usage:
    python generate_ltc_cpp.py --ltc_npy ltc.npy --amp_npy amp.npy --output ltc_table.cpp

The produced .cpp matches the formatting produced by your second script.
"""

import argparse
import numpy as np
import math
import sys

from math import sqrt

def main():
    parser = argparse.ArgumentParser(description='Generate annotated LTC C++ directly from .npy files')
    parser.add_argument('--ltc_npy', type=str, required=True, help='Path to LTC .npy (shape: A x T x 3 x 3)')
    parser.add_argument('--amp_npy', type=str, required=True, help='Path to amp .npy (shape: A x T)')
    parser.add_argument('--output', type=str, required=True, help='Output .cpp file')
    args = parser.parse_args()

    ltc_mat = np.load(args.ltc_npy)
    amp = np.load(args.amp_npy)

    if ltc_mat.ndim != 4 or ltc_mat.shape[2:] != (3, 3):
        raise ValueError(f"Expected ltc_mat shape (A, T, 3, 3), got {ltc_mat.shape}")
    if amp.shape != ltc_mat.shape[:2]:
        raise ValueError(f"Expected amp shape {ltc_mat.shape[:2]}, got {amp.shape}")

    A, T = ltc_mat.shape[0], ltc_mat.shape[1]
    texel_count = A * T
    lut_size = int(round(math.sqrt(texel_count)))
    if lut_size * lut_size != texel_count:
        raise ValueError(f"LUT texel count ({texel_count}) is not a perfect square")

    # Normalize each 3x3 matrix by the length (L2 norm) of its 3rd column (index 2),
    # matching the original pipeline: divide whole matrix by ||col2||
    # shape of third column: (A, T, 3, 1) -> norm over rows (axis=-2) -> (A, T, 1, 1)
    third_col = ltc_mat[:, :, :, 2:3]  # shape (A, T, 3, 1)
    norms = np.linalg.norm(third_col, axis=-2, keepdims=True)  # (A, T, 1, 1)
    # avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    ltc_mat = ltc_mat / norms

    # Collect fields and compute the final four floats per texel directly (F0..F3):
    # M00 = mat[0,0], M01 = mat[0,1], M02 = mat[0,2], etc.
    # normalize by M22 (mat[2,2]) -> divide selected terms by M22
    # F0 = M00 / M22
    # F1 = M02 / M22
    # F2 = M11 / M22
    # F3 = M20 / M22

    # pre-allocate arrays in texel order: alpha major, theta minor
    F0 = np.zeros(texel_count, dtype=np.float32)
    F1 = np.zeros(texel_count, dtype=np.float32)
    F2 = np.zeros(texel_count, dtype=np.float32)
    F3 = np.zeros(texel_count, dtype=np.float32)

    idx = 0
    for a in range(A):
        for t in range(T):
            mat = ltc_mat[a, t]
            M00 = float(mat[0, 0])
            M02 = float(mat[0, 2])
            M11 = float(mat[1, 1])
            M20 = float(mat[2, 0])
            M22 = float(mat[2, 2])

            if M22 == 0.0:
                # fallback to 1.0 to avoid NaNs; warn the user once
                M22 = 1.0

            F0[idx] = M00 / M22
            F1[idx] = M02 / M22
            F2[idx] = M11 / M22
            F3[idx] = M20 / M22
            idx += 1

    # Write annotated C++ directly
    with open(args.output, 'w') as out:
        out.write(f"static const std::array<float4_t, {int(sqrt(texel_count))} * {int(sqrt(texel_count))}> ltc_fit_parameters = {{\n\n")

        for a in range(lut_size):
            alpha_val = max(a / (lut_size - 1), 0.01) if lut_size > 1 else 0.01

            out.write(f"/* ----------------------------------------------------\n")
            out.write(f" *  alpha_idx = {a}   (alpha = {alpha_val:.4f})\n")
            out.write(f" * ---------------------------------------------------- */\n")

            for t in range(lut_size):
                idx = a * lut_size + t
                if (t % 4 == 0):
                    out.write(f"    // theta_idx = {t}\n")
                out.write(f"    make_float4({F0[idx]:.6f}f, {F1[idx]:.6f}f, {F2[idx]:.6f}f, {F3[idx]:.6f}f),\n")

            out.write("\n")

        out.write("};\n\n")

        # ------------------------------------------------------------------
        # Write amplitude LUT
        # ------------------------------------------------------------------

        out.write(f"static const std::array<float, {texel_count}> ggx_conductor_ltc_amplitude_data = {{\n\n")

        for a in range(lut_size):
            alpha_val = max(a / (lut_size - 1), 0.01) if lut_size > 1 else 0.01

            out.write(f"/* ----------------------------------------------------\n")
            out.write(f" *  alpha_idx = {a}   (alpha = {alpha_val:.4f})\n")
            out.write(f" * ---------------------------------------------------- */\n")

            for t in range(lut_size):
                idx = a * lut_size + t
                if (t % 4 == 0):
                    out.write(f"    // theta_idx = {t}\n")
                out.write(f"    {amp[a, t]:.6f}f,\n")

            out.write("\n")

        out.write("};\n")

    print(f"Wrote: {args.output} (LUT size: {lut_size}x{lut_size}, texels: {texel_count})")


if __name__ == '__main__':
    main()
