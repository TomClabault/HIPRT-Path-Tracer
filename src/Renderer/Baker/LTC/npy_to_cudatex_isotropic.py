import numpy as np
import argparse, os, sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ltc_npy', type=str, help='')
    parser.add_argument('--amp_npy', type=str, help='')
    parser.add_argument('--output', type=str, help='Output text file')
    args = parser.parse_args()

    ltc_mat = np.load(args.ltc_npy)
    amp = np.load(args.amp_npy)

    ltc_mat[:, :, :, :] = ltc_mat[:, :, :, :] / np.linalg.norm(ltc_mat[:, :, :, 2:3], axis=-2, keepdims=True)

    f = open(args.output, 'w+')

    f.write('#ifndef LTC_ISO_LUT\n')
    f.write('#define LTC_ISO_LUT\n\n')

    f.write('float ltc_iso_1[%d] = {' % (ltc_mat.shape[0] * ltc_mat.shape[1] * 4))

    for alpha_idx in range(ltc_mat.shape[0]):
        for theta_idx in range(ltc_mat.shape[1]):
            mat = ltc_mat[alpha_idx, theta_idx, :, :]

            f.write('%.8f, ' %  (mat[0, 0]))
            f.write('%.8f, ' %  (mat[0, 1]))
            f.write('%.8f, ' %  (mat[0, 2]))

            if alpha_idx == ltc_mat.shape[0]-1 and theta_idx == ltc_mat.shape[1]-1:
                f.write('0.0f};\n\n')
            else:
                f.write('0.0f, ')
    
    f.write('float ltc_iso_2[%d] = {' % (ltc_mat.shape[0] * ltc_mat.shape[1] * 4))

    for alpha_idx in range(ltc_mat.shape[0]):
        for theta_idx in range(ltc_mat.shape[1]):
            mat = ltc_mat[alpha_idx, theta_idx, :, :]

            f.write('%.8f, ' %  (mat[1, 0]))
            f.write('%.8f, ' %  (mat[1, 1]))
            f.write('%.8f, ' %  (mat[1, 2]))

            if alpha_idx == ltc_mat.shape[0]-1 and theta_idx == ltc_mat.shape[1]-1:
                f.write('0.0f};\n\n')
            else:
                f.write('0.0f, ')
    
    f.write('float ltc_iso_3[%d] = {' % (ltc_mat.shape[0] * ltc_mat.shape[1] * 4))

    for alpha_idx in range(ltc_mat.shape[0]):
        for theta_idx in range(ltc_mat.shape[1]):
            mat = ltc_mat[alpha_idx, theta_idx, :, :]

            f.write('%.8f, ' %  (mat[2, 0]))
            f.write('%.8f, ' %  (mat[2, 1]))
            f.write('%.8f, ' %  (mat[2, 2]))

            if alpha_idx == ltc_mat.shape[0]-1 and theta_idx == ltc_mat.shape[1]-1:
                f.write('%.8f};\n\n' % amp[alpha_idx, theta_idx])
            else:
                f.write('%.8f, ' % amp[alpha_idx, theta_idx])
    
    f.write('#endif')
    f.close()