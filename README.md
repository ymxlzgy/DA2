# DA<sup>2</sup> Dataset: Toward Dexterity-Aware Dual-Arm Grasping

This repo will contain the code for dataset generation and some scripts that can visualize grasp pairs and render virtual scenes. Still under construction.

## Installation
### Basic installation
```
conda create -n DA2 python=3.8
conda activate DA2
git clone https://github.com/ymxlzgy/DA2.git
cd path/to/DA2
pip install -r requirements.txt
```
### Meshpy installation
```
cd path/to/DA2/meshpy
python setup.py develop
```
## Toy generation
This is used to individual generation. Need to modify the path  ``OBJ_FILENAME`` inside ``posetest.py`` to your customized path, and run:
```
cd path/to/DA2/scripts
python posetest.py
```
The generated grasp pairs will be saved under `test_tmp`.
## Whole dataset generation
Need to modify `file_dir` to the customized mesh path.

"Note: `grasp_dir2` is the path that script will save grasp pairs at. 
```
cd path/to/DA2/scripts
python generate_dual_dataset.py
```
The generated grasp pairs will be stored under `grasp2`. 
In case that the generation process is interrupted, you can copy these files to `grasp` and delete the files under `grasp2`. The new generation will skip the files under `grasp`.
`generate_dual_dataset2.py` is used for a parallel generation. If use this script, you need to modify the `len(file_list)` in `generate_dual_dataset.py` to a customized number.
## Visualize
To visualize the mesh with accompanying grasp pairs, run:
```
cd path/to/DA2/scripts
python dual_grasps_visualization.py absolute_path/to/grasp_file --mesh_root path/to/meshes
```


## Render scenes
After download or generating the dataset, run:
```
cd path/to/DA2/scripts
python render_point_dex.py path/to/dataset
```
You may need to change the path under the function **load_grasp_path** to a customized defined path.
The generated scenes will be under ``path/to/dataset/table_scene_stand_all``. Simulated point clouds will be under ``path/to/dataset/pc_two_view_all``, Virtual camera info will be under ``path/to/dataset/cam_pose_all``. 


Scene rendering is time-consuming. To render in parallel, just run multiple scripts in the same time.

If you want to visualize the generated scenes, run:
```
python render_point_dex.py path/to/dataset --load_existing number_of_the_scene --vis
```
## Acknowledgement
This work is based on Dex-Net, Acronym, and Contact-GraspNet.

If you think this repo can help your research, please consider citing:
```javascript
@article{da2dataset,
  author={Zhai, Guangyao and Zheng, Yu and Xu, Ziwei and Kong, Xin and Liu, Yong and Busam, Benjamin and Ren, Yi and Navab, Nassir and Zhang, Zhengyou},
  journal={IEEE Robotics and Automation Letters},
  title={DA$^2$ Dataset: Toward Dexterity-Aware Dual-Arm Grasping},
  year={2022},
  volume={7},
  number={4},
  pages={8941-8948},
  doi={10.1109/LRA.2022.3189959}}
```