# E-HMSNet
 This is the code of our paper "An Extensible Hierarchical Multimodal Semantic Segmentation Network for Underwater Scenarios".
  
# Requirements
  python 3.7/3.8 + pytorch 1.9.0 (built on [EGFNet](https://github.com/ShaohuaDong2021/EGFNet))
   
# Dataset
1. [PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb) [1]
2. UWS dataset

[1] Shivakumar S S, Rodrigues N, Zhou A, et al. Pst900: Rgb-thermal calibration, dataset and segmentation network[C]//2020 IEEE international conference on robotics and automation (ICRA). IEEE, 2020: 9441-9447.


# Training
1. Install '[apex](https://github.com/NVIDIA/apex)'.
2. Run train_ON_PSTNet.py or train_ON_UWS 

# Note
our main model is under './toolbox/models/E_HMSNet.py'

 If you have any questions or suggestions regarding this project, please contact the corresponding author of the paper, Xu Yuezhu (email: xuyuezhu@hrbeu.edu.cn) for further information.


