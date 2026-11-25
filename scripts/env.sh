# source this to use the Puro training environment
source ~/venvs/puro/bin/activate
export CUDA_HOME=$HOME/venvs/puro/cuda_home
export CUDNN_PATH=$CUDA_HOME
export CUDNN_HOME=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
# The compute nodes ship a system CUDA 12 runtime. Transformer Engine refuses to run
# when it can load both libcudart.so.12 and .13, so an empty file of that name earlier
# in the search path makes the CUDA 12 one unloadable for this environment.
export LD_LIBRARY_PATH=$HOME/venvs/puro/block_cuda12:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PURO_MEGATRON_ROOT=$HOME/projects/puro/Puro-Megatron
export PYTHONPATH=$PURO_MEGATRON_ROOT:${PYTHONPATH:-}
