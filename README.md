# cochlea_synapses
![image](https://github.com/ucsdmanorlab/cochlea_synapses/assets/26422897/a3d4cd4b-bf3d-4a07-82d5-b55c807069b0)
Goal: the purpose of this work is to generate segmentation of synapses on hair cells in the cochlea. 
Current training has focused on pre-synaptic (CTBP2) segmentation of inner hair cells only, but can be expanded to include post-synaptic signals and outer hair cells. 
Current training data also includes volumes with varying resolution from varying microscope modalities (laser scanning confocal, spinning disk confocal, and airyscan). 


## install
1) conda create -n synapses python=3.8

2) conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

3) pip install zarr scikit-image neuroglancer imagecodecs tensorboard tensorboardX

4) pip install git+git://github.com/funkey/gunpowder.git

5) pip install git+git://github.com/funkelab/funlib.learn.torch.git

6) pip install git+git://github.com/funkelab/funlib.show.neuroglancer.git
