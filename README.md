# ECaT
Exemplar-based image-to-image translation and cascaded vision transformers for unsupervised multi-modal medical image registration
This is the project page of ECaT.
## ECaT: Exemplar-Based Image-to-Image Translation and Cascaded Vision Transformers for Unsupervised Multi-Modal Medical Image Registration.

We propose a fully unsupervised multi-modal medical image registration method based on image-to-image translation and cascaded vision transformers. We incorporate sample-specific style learning in our framework and achieve significant superior performance over traditional methods.

Demo: (upper) tagging and unregistered cine image sequence; (middle) tagging and fake cine image sequence; tagging and registered cine image sequence. Our method can learn the image style of each frame and boost the registration performance significantly.
<div align=center><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/2_CH_11_15_tag_grid_img.gif"/><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/3_CH_12_16_tag_grid_img.gif"/><img width="200" height="400" src="https://github.com/DeepTag/cardiac_tagging_motion_estimation/blob/main/figures/4_CH_10_14_tag_grid_img.gif"/></div>

## Acknowledgments
Our code implementation borrows heavily from [F-LSeSim](https://github.com/lyndonzheng/F-LSeSim).
