# KDD-2023-CriticalFL
Presented here are the foundational algorithm implementations within the context of Federated Learning (FL). This resource encompasses the key components outlined in our paper.

# Usage 
1. Requirement: Ubuntu 20.04, Python v3.5+, Pytorch and CUDA environment
2. "./Main.py" is about configurations and the basic Federated Learning framework
3. "./Sims.py" describes the simulators for clients and central server
4. "./Utils.py" contains all necessary functions and discusses how to get training and testing data
5. "./Settings.py" describes the necessary packages and settings
6. Folder "./Models" includes codes for AlexNet, VGG-11, ResNet-18 and LSTM
7. Folder "./Optim" includes codes for FedProx, VRL-SGD, FedNova
8. Folder "./Comp_FIM" is the library to calculate Fisher Information Matrix (FIM)

# Implementation
 1. Use "./Main.py" to run results, the command is '''python3 ./Main.py'''
 2. Parameters can be configured in "./Main.py"
```
  Configs['dname'] = "cifar10"
  Configs["mname"] = "alex"
  Configs['nclients'] = 128
  Configs['pclients'] = 16
  Configs["learning_rate"] = 0.01
```

# Citation
If you use the simulator or some results in our paper for a published project, please cite our work by using the following bibtex entry

```
@inproceedings{yan2023criticalfl,
  title={CriticalFL: A Critical Learning Periods Augmented Client Selection Framework for Efficient Federated Learning},
  author={Gang Yan, Hao Wang, Xu Yuan and Jian Li},
  booktitle={Proc. of ACM SIGKDD},
  year={2023}
}
```

