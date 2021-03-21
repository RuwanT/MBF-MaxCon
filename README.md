# Consensus Maximisation Using Influences of Monotone Boolean Functions

Code for paper "Consensus Maximisation Using Influences of Monotone Boolean Functions" to be presented at CVPR 2021 - oral presentation. 


## Installation
The code was tested on a macOS Catalina and Ubuntu 16.04 with MATLAB 2019b. Requires MATLAB communications toolbox.  
1) Install `VlFeat` (https://www.vlfeat.org/install-matlab.html)  
2) Install [SeDuMi: Optimization over symmeric cones](https://github.com/sqlp/sedumi).  This is required for A*.  

		* Download sedumi from the above URL.  
		* Copy sedumi folder in to folder linearASTAR.   
		* run script `install_sedumi.m`  



## Note
Please note that in the paper the Feasibility/Infeasibility function is represented as <img src="https://render.githubusercontent.com/render/math?math=f\{0,1\}^n \to \{0,1\}"> wheras in the code the function is represented as <img src="https://render.githubusercontent.com/render/math?math=f\{0,1\}^n \to \{1,-1\}">. Where <img src="https://render.githubusercontent.com/render/math?math=f(x) = -1"> means Infeasible.



## Running the code

### Simple Example - MBF-MaxCon
Two dimentional linear fitting with synthetic data 
> Run `MaxConMBF_simple_example.m`


### Synthetic data experiments - MaxCon
Eight dim linear fitting with synthetic data - comparison and ablation studies
> Run `maxcon_linear_demo.m`



###  Linear Fundamental Matrix Estimation - MaxCon
> Run `maxcon_linear_fundamental.m`


### Synthetic data experiments - Fourier Calculations
Calculate Fourier coefficients for a toy 2D line fitting problem using different sampling methods: "Exact", "Uniform sampling", "Goldreich-Levin", "MBF-ODonnell-2005"
> Run `demo_linear.m` in MBF_basics folder

Calculate the error in influence estimation
> Comparison between "uniform-sampling" and "exact" influences on a toy 2D line fitting problems Run `influence_est_accuracy.m` in MBF_basics folder






## Code Reference

If you find this work useful in your research, please consider [citing](https://arxiv.org/abs/2103.04200):

```
@article{tennakoon2021consensus,
  title={Consensus Maximisation Using Influences of Monotone Boolean Functions},
  author={Tennakoon, Ruwan and Suter, David and Zhang, Erchuan and Chin, Tat-Jun and Bab-Hadiashar, Alireza},
  journal={arXiv preprint arXiv:2103.04200},
  year={2021}
}
```



### ASTAR code is from [[Github Page](https://github.com/ZhipengCai/MaxConTreeSearch.git)] 

Please acknowledge the original authors by citing in any academic publications that have made use of this package or part of it:

```
@InProceedings{Cai_2019_ICCV,
author = {Cai, Zhipeng and Chin, Tat-Jun and Koltun, Vladlen},
title = {Consensus Maximization Tree Search Revisited},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
year = {2019}
}
```

### RANSAC code is inspired by [[Github Page](https://github.com/ZhipengCai/Demo---Deterministic-consensus-maximization-with-biconvex-programming.git)] 

<!--Please acknowledge the original authors by citing in any academic publications that have made use of this package or part of it:

```
@inproceedings{cai2018deterministic,
  title={Deterministic Consensus Maximization with Biconvex Programming},
  author={Cai, Zhipeng and Chin, Tat-Jun and Le, Huu and Suter, David},
  booktitle={European Conference on Computer Vision},
  pages={699--714},
  year={2018},
  organization={Springer}
}
```
-->


