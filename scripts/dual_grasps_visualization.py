#!/usr/bin/env python3
"""
Author: Guangyao Zhai

"""

import sys
import json
import trimesh
import argparse
import numpy as np
import copy

from DA2_tools import load_mesh, load_dual_grasps, create_robotiq_marker
from autolab_core import RigidTransform


def make_parser():
    parser = argparse.ArgumentParser(
        description="Visualize grasp pairs from the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="HDF5 or JSON Grasp file(s).")
    parser.add_argument(
        "--num_grasps", type=int, default=50, help="Number of grasps to show."
    )
    parser.add_argument(
        "--mesh_root", default=".", help="Directory used for loading meshes."
    )
    parser.add_argument(
        "--scale", type=float, default=1
    )
    parser.add_argument(
        "--metric", type=str, default='for', help="for/tor/dex."
    )
 
    parser.add_argument(
        "--quality", type=float, default=0.5, help="visualize grasp pairs having larger quality than this."
    )
    return parser

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x).all() :
            count = count + 1
    return count

def main(argv=sys.argv[1:]):
    parser = make_parser()
    args = parser.parse_args(argv)
    successful_grasps = []
    marker = []
    database = []
    wave = args.num_grasps//3

    for f in args.input:
        # load object mesh
        obj_mesh = load_mesh(f, mesh_root_dir=args.mesh_root)

        T, f, d, t = load_dual_grasps(f)
        if args.metric == 'for':
            metric = f
        elif args.metric == 'dex':
            metric = d
        elif args.metric == 'tor':
            metric = t
        
        if T.size == 0:
            obj_mesh.show()
        elif args.num_grasps == 1:
            t = T[np.random.choice(np.where((metric >= args.quality))[0], 1)]
            trimesh.Scene([obj_mesh] + [create_robotiq_marker(color=[255, 0, 0]).apply_transform(t[0][0])] + [create_robotiq_marker(color=[255, 0, 0]).apply_transform(t[0][1])]).show()
        else:
            print(np.where( (metric >= args.quality))[0])
            for i, (t1, t2) in enumerate(T[np.random.choice(np.where( (metric >= args.quality))[0], args.num_grasps) if len(np.where( (metric >= args.quality))[0]) > args.num_grasps else np.where( (metric >= args.quality))[0]]):

                current_t1 = countX(database, t1)
                current_t2 = countX(database, t2)
                color = i/wave*255
                code1 = color if color<=255 else 0
                code2 = color%255 if color>255 and color<=510 else 0
                code3 = color%510 if color>510 and color<=765 else 0
                successful_grasps.append((create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t1), create_robotiq_marker(color=[code1, code2, code3]).apply_transform(t2)))

                trans1 = t1.dot(np.array([0,-0.067500/2-0.02*current_t1,0,1]).reshape(-1,1))[0:3]
                trans2 = t2.dot(np.array([0,-0.067500/2-0.02*current_t2,0,1]).reshape(-1,1))[0:3]

                tmp1 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans1).matrix)
                tmp1.visual.face_colors = [code1, code2, code3]
                tmp2 = trimesh.creation.icosphere(radius = 0.01).apply_transform(RigidTransform(np.eye(3), trans2).matrix)
                tmp2.visual.face_colors = [code1, code2, code3]
                marker.append(copy.deepcopy(tmp1))
                marker.append(copy.deepcopy(tmp2))
                database.append(t1)
                database.append(t2)


            trimesh.Scene([obj_mesh] + successful_grasps + marker).show()



if __name__ == "__main__":
    main()
