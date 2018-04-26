# Precipitation-Nowcasting

This is an easy-to-understand implementation of ConvLSTM model(fisrt proposed by [Xinjian Shi et al.])(https://arxiv.org/abs/1506.04214https://arxiv.org/abs/1506.04214) in a real-world precipitation nowcasting problem with Pytorch. Here presents the guidance on how to run this project by yourself. Have fun!

## DATA
#### Two open-sourced datasets are available for training and testing in this project.

1. A pre-masked radar datasets.(Included in the package)     
2. Tianchi CNKI 2017 dataset（Provided by Shenzhen Meteorological Bureau）.This dataset is not included yet. However, You can download the datasets [here](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.6d453864enogCW&raceId=231596)

## Getting Started
### Prerequisites  
Environment:   
* Ubuntu 16.04+   
* Anaconda 3-5.1  
* Python 3.6  
* CUDA 8
     
Python 3.6 Packages needed:  
&ensp;&ensp;`Arrow`
&ensp;&ensp;`Pytorch 0.3.1` 
&ensp;&ensp;`PIL`

### Installing
1. Install CUDA8
```
  \# sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
  \# sudo apt-get update
  \# sudo apt-get install cuda
```
2. Download and install Anaconda environment 
```
  \# wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
  \# bash Anaconda3-5.1.0-Linux-x86_64.sh
```
&ensp;&ensp;&ensp;&ensp;More on how to install Anaconda, see [`this passage`](https://www.jianshu.com/p/03d757283339)  

3. Install an environment(optional but suggested)
```
  \# conda create -n pytorch python=3.6 
```
4. Activate your new-built environemt and install Pytorch and torchvision
```
  \# source activate pytorch 
  \# pip3 install torch torchvision
```

### Train the model 

1. Download the all package and unpack it with the command:  
``` 
  \# tar zxvf conlstm.tar.gz YOUR WORK PATH
```  
Note: you also need to unpack the files in the original `data` directory before training  

2. Train the model 
```
  Python3 training.py
```

### Running the test 

Run the test.py with the command. 
```
  python3 test.py  
```
Evaluate your model's performance by running 
```
  python3 evaluate.py
```

### Authors  
     cxxixi
     pqx

## Notes
1. [`Notes on ConvLSTM`](https://github.com/cxxixi/Precipitation-Nowcasting/issues/1)
