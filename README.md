# Precipitation-Nowcasting

This is an easy-to-understand implementation of ConvLSTM model(fisrt proposed by [Xinjian Shi et al.])(https://arxiv.org/abs/1506.04214https://arxiv.org/abs/1506.04214) in a real-world precipitation nowcasting problem with Pytorch. Here presents the guidance on how to run this project by yourself. Enjoy!

## DATA
##### Two open-sourced datasets are available for training and testing in this project.

1. An pre-masked radar datasets.(Included in the package)   
2. Tianchi CNKI 2017 dataset（Provided by Shenzhen Meteorological Bureau）.This dataset is not included yet. However, You can download the datasets [here](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.6d453864enogCW&raceId=231596)

## Getting Started
### Prerequisites  
Environment:   
     Ubuntu 16.04+ 
     Anaconda ....
     Python 3.6  
     Pytorch 0.3.1  
     
Python 3.6 Packages needed:  
     Arrow 
    
### Installing

1. [`Download the all package`]() and unpack it with the command:
[```

```
2. 
```

```

### Running the tests  
Run the test.py with the command. 
```
python3 test.py  
```
You'll get a visualization of the CSI, POD, FAR like this:  
(CSI: successful )



## Notes
1. [`Notes on ConvLSTM`](https://github.com/cxxixi/Precipitation-Nowcasting/issues/1)

