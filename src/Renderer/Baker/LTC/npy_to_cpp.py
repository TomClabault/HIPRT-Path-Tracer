import numpy as np
import argparse, os, sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--npy', type=str, help='')
    parser.add_argument('--output', type=str, help='Output text file')
    args = parser.parse_args()

    ltc_mat = np.load(args.npy)
    ltc_mat[:, :, :, :, :, :] = ltc_mat[:, :, :, :, :, :] / np.linalg.norm(ltc_mat[:, :, :, :, :, 2:3], axis=-2, keepdims=True)
    f = open(args.output, 'w+')

    f.write('float anisomats[%d][%d][%d][%d][%d] = {\n' % (ltc_mat.shape[0], ltc_mat.shape[1], ltc_mat.shape[2], ltc_mat.shape[3], 9))

    for alphax_idx in range(ltc_mat.shape[0]):
        f.write('{')
        for alphay_idx in range(ltc_mat.shape[1]):
            f.write('{')
            for theta_idx in range(ltc_mat.shape[2]):
                f.write('{')
                for phi_idx in range(ltc_mat.shape[3]):
                    f.write('{')

                    mat = ltc_mat[alphax_idx, alphay_idx, theta_idx, phi_idx, :, :]

                    f.write('%.8f, ' %  (mat[0, 0]))
                    f.write('%.8f, ' %  (mat[0, 1]))
                    f.write('%.8f, ' %  (mat[0, 2]))
                    f.write('%.8f, ' %  (mat[1, 0]))
                    f.write('%.8f, ' %  (mat[1, 1]))
                    f.write('%.8f, ' %  (mat[1, 2]))
                    f.write('%.8f, ' %  (mat[2, 0]))
                    f.write('%.8f, ' %  (mat[2, 1]))

                    if phi_idx == ltc_mat.shape[3]-1:
                        f.write('%.8f}\n' %  (mat[2, 2]))
                    else:
                        f.write('%.8f},\n' %  (mat[2, 2]))
                
                if theta_idx == ltc_mat.shape[2]-1:
                    f.write('}')
                else:
                    f.write('},')

            if alphay_idx == ltc_mat.shape[1]-1:
                f.write('}')
            else:
                f.write('},')

        if alphax_idx == ltc_mat.shape[0]-1:
            f.write('}')
        else:
            f.write('},')
    
    f.write('};')
    
    f.close()