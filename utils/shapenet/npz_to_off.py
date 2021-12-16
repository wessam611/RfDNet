import argparse
import os

import numpy as np
import common
# from common import write_off

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--catid', type=str, required=True, dest='catid')
    ap.add_argument('--shapeid', type=str, required=True, dest='shapeid')
    ap.add_argument('--out_file', type=str, required=True, dest='out_file')

    args = ap.parse_args()
    path = os.path.join('datasets', 'ShapeNetv2_data', 'pointcloud', args.catid, f'{args.shapeid}.npz')
    
    point_dict = np.load(path)
    pointcloud = point_dict['points']

    tuples = []
    for i in range(pointcloud.shape[0]):
        tuples.append(tuple(pointcloud[i]))

    common.write_off(args.out_file, tuples, [])

if __name__ == '__main__':
    main()