The Griewank function approximated by a neural network network.
- Written using pytorch
- Uses 4 GPUs on single node using the native pytorch distribution

Sources
 - source: https://towardsdatascience.com/pytorch-tabular-regression-428e9c9ac93
 - source: https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#create-model-and-dataparallel

Slurm Commands
...
 - module load Anaconda3/5.3.0
 - conda create -n pytorch
 - source activate pytorch
 - conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch  #this only the first time
 - salloc -N1 --ntasks-per-node=4 --gres=gpu:4 
 - python pytorch_deep_regr_mult_gpu_mult_nodes.py
