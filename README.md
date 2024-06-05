# cochlea_synapses
![snapses](https://github.com/ucsdmanorlab/cochlea_synapses/assets/26422897/6f28028c-9e64-45a0-97c2-ddce3b5def3b)

Goal: the purpose of this work is to generate 3d segmentation of synapses on hair cells in the cochlea. 
Current training has focused on pre-synaptic (CTBP2) segmentation of inner hair cells only, but can be expanded to include post-synaptic signals and outer hair cells. 
Current training data also includes volumes with varying resolution from varying microscope modalities (laser scanning confocal, spinning disk confocal, and airyscan). 

## workflow:
1. Data conversion: data is converted to zarr using scripts in 01_data
2. Training and prediction: training and prediction scripts for 2D, 2.5D, and 3D models in 02_train. (Validation statistics also recorded at time of prediction.)

## install
1) conda create -n synapses python=3.9

2) conda activate synapses

3) conda install pytorch=1.12 torchvision torchaudio cudatoolkit=10.2 -c pytorch

4) python -m pip install zarr scikit-image neuroglancer imagecodecs tensorboard tensorboardX

5) python -m pip install git+https://github.com/funkey/gunpowder.git

6) python -m pip install git+https://github.com/funkelab/funlib.learn.torch.git

7) python -m pip install git+https://github.com/funkelab/funlib.show.neuroglancer.git
