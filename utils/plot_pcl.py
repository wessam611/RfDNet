import os
import argparse

import numpy as np
import trimesh

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', type=str)
    ap.add_argument('index', type=int)
    args = ap.parse_args()

    p = os.path.abspath(args.path)
    pcls = np.load(p)
    pcl = pcls[f'arr_{args.index}']

    points = trimesh.points.PointCloud(pcl)
    points.show()

if __name__ == '__main__':
    main()