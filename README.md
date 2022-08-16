## ECaT: Exemplar-Based Image-to-Image Translation and Cascaded Vision Transformers for Unsupervised Multi-Modal Medical Image Registration.

We propose a fully unsupervised multi-modal medical image registration method based on image-to-image translation and cascaded vision transformers. We incorporate sample-specific style learning and hybrid global affine and local nonrigid deformation estimation in our framework and achieve significant superior performance over traditional methods.
<div align=center><img width="820" height="358" src="https://github.com/DeepTag/ECaT/blob/main/f2p.png"/></div>

## Demo
We aim to register an untagged cine MR (cMR) image to a tagged MR (tMR) image. The non-smooth contrast change between tMR and cMR, i.e., presence of tags, poses a great challenge to this task. We use the proposed method to first translate tMR to fake cMR; then, we register cMR to fake cMR using the NCC dissimilarity loss.  (upper) tMR and unregistered cMR sequence; (middle) tMR and fake cMR sequence; (bottom) tMR and registered cMR sequence. Our method can learn the specific image style of each cMR frame to be registered and boost the registration performance significantly. Note how the fake cMR frames can capture the individual image styles of the corresponding real cMR frames.
<div align=center><img width="620" height="620" src="https://github.com/DeepTag/ECaT/blob/main/tfc.gif"/></div>

## Acknowledgments
Our code implementation borrows heavily from [F-LSeSim](https://github.com/lyndonzheng/F-LSeSim), [C2FViT](https://github.com/cwmok/C2FViT), and [Deep-Exemplar-based-Video-Colorization](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization).
