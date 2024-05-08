# cochlea_synapses
![snapses](https://github.com/ucsdmanorlab/cochlea_synapses/assets/26422897/6f28028c-9e64-45a0-97c2-ddce3b5def3b)

Goal: the purpose of this work is to generate 3d segmentation of synapses on hair cells in the cochlea. 
Current training has focused on pre-synaptic (CTBP2) segmentation of inner hair cells only, but can be expanded to include post-synaptic signals and outer hair cells. 
Current training data also includes volumes with varying resolution from varying microscope modalities (laser scanning confocal, spinning disk confocal, and airyscan). 

## workflow:
1. Data conversion: data is converted to zarr using scripts in 01_data
2. Training and prediction: training and prediction scripts for 2D, 2.5D, and 3D models in 02_train. (Validation statistics also recorded at time of prediction.)

## install
1) conda create -n synapses python=3.8

2) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

3) pip install zarr scikit-image neuroglancer imagecodecs tensorboard tensorboardX

4) pip install git+git://github.com/funkey/gunpowder.git

5) pip install git+git://github.com/funkelab/funlib.learn.torch.git

6) pip install git+git://github.com/funkelab/funlib.show.neuroglancer.git
