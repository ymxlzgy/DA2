# DA<sup>2</sup> Dataset: Toward Dexterity-Aware Dual-Arm Grasping

The project website is https://sites.google.com/view/da2dataset. This repo contains the code for DA<sup>2</sup> dataset generation and some scripts that can visualize grasp pairs and render virtual scenes. The paper is available [here](https://arxiv.org/pdf/2208.00408.pdf).

## Installation
### Basic installation
```
conda create -n DA2 python=3.8
conda activate DA2
git clone https://github.com/ymxlzgy/DA2.git
cd path/to/DA2
mkdir grasp test_tmp
pip install -r requirements.txt
```
### Meshpy installation
```
cd path/to/DA2/meshpy
python setup.py develop
```
### Pytorch installation
Please refer to [pytorch](https://pytorch.org/) official website to find the best version in your case, e.g.,:
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
### Mayavi installation
```
conda install mayavi -c conda-forge
```

## Mesh Download
Download meshes [here](https://drive.google.com/file/d/1sc9gTAmkb2VDfn4XJqgpObZW1ZXYckRw/view). After download, unzip them under `path/to/DA2` make it like `path/to/DA2/simplified`.

## Toy generation
This is used to individual generation. Need to modify the path  ``OBJ_FILENAME`` inside ``posetest.py`` to your customized path, and run:
```
cd path/to/DA2/scripts
python posetest.py
```
The generated grasp pairs will be saved under `test_tmp`.
## Whole dataset generation
Need to modify `file_dir` to the customized mesh path.

```
cd path/to/DA2/scripts
python generate_dual_dataset.py
```
The generated grasp pairs will be stored under `grasp`. 
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
`path/to/dataset` in our case is `path/to/DA2`. You may need to change the path under the function **load_grasp_path** to a customized defined path.
The generated scenes will be under ``path/to/dataset/table_scene_stand_all``. Simulated point clouds will be under ``path/to/dataset/pc_two_view_all``, Virtual camera info will be under ``path/to/dataset/cam_pose_all``. 


Scene rendering is time-consuming. To render in parallel, just run multiple scripts in the same time.

If you want to visualize the generated scenes, run:
```
python render_point_dex.py path/to/dataset --load_existing number_of_the_scene --vis
```
## Acknowledgement
This work is based on [Dex-Net](https://github.com/BerkeleyAutomation/dex-net), [Acronym](https://github.com/NVlabs/acronym), [Contact-GraspNet](https://github.com/NVlabs/contact_graspnet), and [diverse-and-stable-grasp](https://github.com/tengyu-liu/diverse-and-stable-grasp).

Many functions under [dexnet](https://github.com/ymxlzgy/DA2/tree/main/dexnet) are from [Dex-Net](https://github.com/BerkeleyAutomation/dex-net). Didn't remove them in case anyone can notice them and may facilitate one's research.

If you think this repo can help your research, please consider citing:
```
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