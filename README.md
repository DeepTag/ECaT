## ECaT: Exemplar-Based Image-to-Image Translation and Cascaded Vision Transformers for Unsupervised Multi-Modal Medical Image Registration.

We propose a fully unsupervised multi-modal medical image registration method based on image-to-image translation and cascaded vision transformers. We incorporate sample-specific style learning and hybrid global affine and local nonrigid deformation estimation in our framework and achieve significant superior performance over traditional methods.
<div align=center><img width="750" height="349" src="https://github.com/DeepTag/ECaT/blob/main/ecat.png"/></div>

Demo: (upper) tagging and unregistered cine sequence; (middle) tagging and fake cine sequence; (bottom) tagging and registered cine sequence. Our method can learn the specific image style of each frame and boost the registration performance significantly.
<div align=center><img width="450" height="450" src="https://github.com/DeepTag/ECaT/blob/main/tfc.gif"/></div>

## Acknowledgments
Our code implementation borrows heavily from [F-LSeSim](https://github.com/lyndonzheng/F-LSeSim), [C2FViT](https://github.com/cwmok/C2FViT), and [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization).
