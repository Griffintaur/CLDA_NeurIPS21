## CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation [[Paper]](https://arxiv.org/abs/2107.00085) [[Website]](https://griffintaur.github.io/CLDA_NeurIPS/)

This repository contains the implementation details of our CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation (CLDA) approach for domain adaptation in images.

Ankit Singh , "CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation"

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{
singh2021clda,
title={{CLDA}: Contrastive Learning for Semi-Supervised Domain Adaptation},
author={Ankit Singh},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=1ODSsnoMBav}
}
```

## Acknowledgments
The implementation is built on the pytorch implementation of [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) and [APE](https://github.com/TKKim93/APE)

### Dataset Structure
You can download the datasets by following the instructions in [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME).
```
data---
     |
   multi---
     |   |
     |  Real
     |  Clipart
     |  Product
     |  Real
   office_home---
     |         |
     |        Art
     |        Clipart
     |        Product
     |        Real
   office---
     |    |
     |   amazon
     |   dslr
     |   webcam
   txt---
       | 
      multi---
       |    |
       |   labeled_source_images_real.txt
       |   unlabeled_target_images_real_3.txt
       |   labeled_target_images_real_3.txt         
       |   unlabeled_source_images_sketch.txt
       |                  ...
      office---
       |     |
       |   labeled_source_images_amazon.txt
       |   unlabeled_target_images_amazon_3.txt
       |   labeled_target_images_amazon_3.txt         
       |   unlabeled_source_images_webcam.txt
       |                  ...
      office_home---
                  |
                 ...       
```

### Example
#### Train
* DomainNet (clipart, painting, real, sketch)
```
python clda_final.py --dataset multi --source real --target sketch --save_interval 500 --steps 50000 --net resnet34 --num 3 --save_check
```
* Office-home (Art, Clipart, Product, Real)
* Office (amazon, dslr, webcam)

