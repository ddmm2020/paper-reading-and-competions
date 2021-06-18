# paper-reading-and-competions

# competions
# Cassava Leaf Disease Classification 【bronze medal】  
Identify the type of disease present on a Cassava Leaf image  
From：Makerere University AI Lab  
### Summary
1. Try different loss functions:  
Basic loss function for classification is cross-entropy loss.But because of the noisy label in this competion, the loss functions with high tolerance to noisy label has better performance.some loss function from ***top solutions***  as below:  
    * Bi-Tempered Logistic Loss (NIPS 2019)  
    * Taylor Cross Entropy loss (IJCAI 2020)  
    * label smoothing is a useful trick.  
2. ViT and EfficientNet are better backbones.  
    Attention Learning in CV.
4. Ensemble
Injecting a lot of diversity in the ensemble is the key to prevent shake in the Private Dataset.
    * light or heaevy TTA is useful
    * stacking 
    * bagging
    * Ensemble different backbones
4. Confident Learning (ICML 2020) [paper](https://arxiv.org/pdf/1911.00068.pdf)  
    Cleanlab package can help us to find some error label.
    
5. Augmentations:  
 Includeing standard enhancements below mix is useful:  
 Mixup(ICLR 2018) [paper](https://arxiv.org/pdf/1710.09412.pdf)/Cutmix(ICCV 2019) [paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf)  /SnapMix(AAAI 2021) [paper](https://arxiv.org/pdf/2012.04846.pdf)

  ![snapMix](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/competions/images/snapMix.png)
  
7. Others
    * Advprop(CVPR 2020) [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xie_Adversarial_Examples_Improve_Image_Recognition_CVPR_2020_paper.pdf): 
        
        Using auxiliary BN to improve model performance with adversarial examples.
    * LeafGAN(T-ASE 2020) [paper](https://arxiv.org/pdf/2002.10100.pdf): 
        
        attention + CycleGAN to generate Leaf images.
    * CAM [paper](https://arxiv.org/pdf/1512.04150.pdf)：
        
        Help to understand which pixels features are important.

[Some Code and More details](https://github.com/ddmm2020/paper-reading-and-competions/blob/main/competions)


