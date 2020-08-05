# DDA
Discriminative distribution alignment: A uniﬁed framework for heterogeneous domain adaptation. Pattern Recognition 2020.

# Datasets

You can download the sample datasets in:

Link: https://pan.baidu.com/s/1T2zSSz8mlROrBmclOk3kwA

Password: lqwt

# Running

1. You can run the two codes by performing the file of Main.m. The results of DDACL and DDASL should be close to 91.57 (SCS->TAD) and 90.93 (SCS->TAD), respectively. Note that different environmental outputs may be different.

2. You can use your datasets by replacing: 

   ```
   source_exp = {SCS} % source domain data 
   target_exp = {TAD} % target domain data
   ```

3. You can tune the parameters, i.e., beta, tau, lambda, d, T, for different applications.

4. The default parameters are: beta = 0.001, tau = 0.002, lambda = 0.001, d = 100, T = 5.

# Citation
If you find it is helpful, please cite:

```
@article{Yao-2020,
  author    = {Yuan Yao and Yu Zhang and Xutao Li and Yunming Ye},
  title     = {Discriminative distribution alignment: {A} unified framework for heterogeneous domain adaptation},
  journal   = {Pattern Recognit.},
  volume    = {101},
  pages     = {107165},
  year      = {2020},
}
```
