在你的虛擬主機上建立.sif檔案
[user@localhost ]$ vi  h2o4gpuPy.def
BootStrap: docker
From: nvidia/cuda:12.1.0-devel-ubuntu18.04

# Note: This container will have only the Python API enabled

%environment
# -----------------------------------------------------------------------------------
    export PYTHON_VERSION=3.6
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export LC_ALL=C

%post
# -----------------------------------------------------------------------------------
# this will install all necessary packages and prepare the contianer

    export PYTHON_VERSION=3.6
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64/:$CUDA_HOME/lib/:$CUDA_HOME/extras/CUPTI/lib64

    apt-get -y update
    apt-get install -y --no-install-recommends build-essential
    apt-get install -y --no-install-recommends git
    apt-get install -y --no-install-recommends vim
    apt-get install -y --no-install-recommends wget
    apt-get install -y --no-install-recommends ca-certificates
    apt-get install -y --no-install-recommends libjpeg-dev
    apt-get install -y --no-install-recommends libpng-dev
    apt-get install -y --no-install-recommends libpython3.6-dev
    apt-get install -y --no-install-recommends libopenblas-dev pbzip2
    apt-get install -y --no-install-recommends libcurl4-openssl-dev libssl-dev libxml2-dev
    apt-get install -y --no-install-recommends python3-pip
    apt-get install -y --no-install-recommends wget

    ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
    ln -s /usr/bin/pip3 /usr/bin/pip

    pip3 install setuptools
    pip3 install --upgrade pip

    wget https://s3.amazonaws.com/h2o-release/h2o4gpu/releases/stable/ai/h2o/h2o4gpu/0.4-cuda10/rel-0.4.0/h2o4gpu-0.4.0-cp36-cp36m-linux_x86_64.whl
    pip install h2o4gpu-0.4.0-cp36-cp36m-linux_x86_64.whl
    
[user@localhost ]$ singularity build --fakeroot h2o4gpuPy11.sif h2o4gpuPy.def
INFO:    Starting build...
...
INFO:    Adding environment to container
INFO:    Creating SIF file...
INFO:    Build complete: h2o4gpuPy.sif
[user@localhost ]$ ls
h2o4gpuPy.def  h2o4gpuPy.sif  

# Upload to server
[user@localhost ]$ sftp user@140.110.148.5
(user@140.110.148.5) Please select the 2FA login method.
1. Mobile APP OTP
2. Mobile APP PUSH
3. Email OTP
Login method: 2                      # two factor
(user@140.110.148.5) Password:   # input password
Connected to 140.110.148.5.
sftp> put h2o4gpuPy.sif              #上傳檔案(h2o4gpuPy.sif)
Uploading h2o4gpuPy.sif to /home/user/h2o4gpuPy.sif
h2o4gpuPy.sif                                                    100% 3503MB 243.4MB/s   00:14
sftp> exit
[user@localhost ]$
