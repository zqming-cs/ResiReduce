# ResiReduce: Saving Memory via Residual Reduction for DNN Training with Compressed Communication

__ResiReduce__ is  a memory-saving mechanism that reuses residuals across similar layers and applies strategic compression within specific layers. Experiments on local and cloud clusters show that __ResiReduce__ can reduce the memory footprint of the model states by up to 15.7% while preserving the model accuracy and training throughput.


# Introduction
This code repository covers:
### __ResiReduce Framework__
- __ResiReduce__(Naive): Leverage residual reduction to save GPU memory for DNN training
- __ResiReduce__-Inter: Inter-layer residual reduction schemes
- __ResiReduce__-Intra: Intra-layer residual reduction schemes



### __State-of-the-art sparsification algorithms__

- [DGC](https://arxiv.org/pdf/1712.01887.pdf)
- [Gaussiank](https://arxiv.org/pdf/1911.08772.pdf)
- [Redsync](https://www.sciencedirect.com/science/article/pii/S0743731518308657)
- [SIDCo](https://proceedings.mlsys.org/paper_files/paper/2021/file/fea47a8aa372e42f3c84327aec9506cf-Paper.pdf)

# Implementation



## **__ResiReduce__** System Architecture
We use the [PyTorch](https://github.com/pytorch/pytorch) framework and implemented the prototype system of __ResiReduce__ based on the [Horovod](https://github.com/horovod/horovod) distributed training framework using NCCL as the communication library.
<!-- The overview of our system is as follows:  -->
<!-- ![Overview](Overview.png) -->
<!-- <center class ='img'>
<img src="Overview_.png" width="600px" />
</center> -->


In our system of __ResiReduce__, We first leverage residual reduction to save GPU memory for DNN training. We design ResiReduce composed of (i) two inter-layer residual reduction
schemes (a straightforward one and an improved one) based on the similarity of adjacent residuals and (ii) two intra-layer residual reduction schemes (a naive one and an improved one) by an L1-norm based strategic compression.

<!-- ## **__ResiReduce__** Generator Workflow
The workflow of the __ResiReduce__ __Generator__ module：
<center class ='img'>
<img src="Generator_.png" width="600px" />
</center> -->

# Installation


## **Prerequisites**
- CUDA-12.0
- Python >= 3.9
- [NCCL-2.8.3](https://github.com/NVIDIA/nccl)
- [PyTorch-1.3.+](https://github.com/pytorch/pytorch)
- [OpenMPI-4.0.+](https://www-lb.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.28.1+](https://github.com/horovod/horovod)
- [Numpy](https://github.com/numpy/numpy)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Tqdm](https://github.com/tqdm/tqdm)

## **Get the code**
```
git clone https://github.com/zqming-cs/ResiReduce.git
cd ResiReduce
pip install -r requirements.txt
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.28.0
```

if pip installation fails, please try to upgrade pip via `pip install --upgrade pip`. If [Horovod](https://github.com/horovod/horovod) installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html).

## **Quick start**
The primary benchmark is provided in `example`. 
For example, we can use the following command to run the benchmark on 8 GPUs, with compression algorithm as dgc, communication primitive as allgather, memory as residual.
 
**To run BERT-large training job:**
```
cd ResiReduce/example/nlp/bert/scripts
bash run_squad_bert.sh
```

**To run GPT2-large training job:**
```
cd ResiReduce/example/nlp/gpt
bash run_clm_no_trainer_hvd_103.sh
```

**To run ViT-large training job:**
```
cd ResiReduce/example/cv/vit
bash run_imagenet_no_trainer.sh
```

**To run ResNet-152 training job:**
```
cd ResiReduce/example/cv/resnet
bash run_imagenet_resnet152.sh
```


## **Papers**

ResiReduce: Saving Memory via Residual Reduction for DNN Training with Compressed Communication

If you are using this repository for your paper, please cite our work
```
@inproceedings{zheng2025saving,
  title={Saving Memory via Residual Reduction for DNN Training with Compressed Communication},
  author={Zheng, Xinjue and Ming, Zhangqiang and Hu, Yuchong and Yao, Chenxuan and Zhou, Wenxiang and Wang, Rui and Chen, Xun and Feng, Dan},
  booktitle={European Conference on Parallel Processing},
  pages={221--235},
  url={https://doi.org/10.3724/SP.J.1089.2022.18852}
  year={2025},
  organization={Springer}
}
```

## **Referred Datasets**

- Wikitex-2/103: [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)
- SQuAD: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
- CIFAR-100: [https://www.cs.utoronto.ca/~kriz/cifar.html](https://www.cs.utoronto.ca/~kriz/cifar.html)
- ImageNet: [https://www.image-net.org/](https://www.image-net.org/)

## **License**

See [LICENSE](https://github.com/zqming-cs/ResiReduce/blob/main/LICENSE.txt).
