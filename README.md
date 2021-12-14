# cochlea_synapses

conda create -n synapses python=3.8

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

pip install zarr scikit-image neuroglancer imagecodecs tensorboard tensorboardX

pip install git+git://github.com/funkey/gunpowder.git
pip install git+git://github.com/funkelab/funlib.learn.torch.git
pip install git+git://github.com/funkelab/funlib.show.neuroglancer.git

