---
license: mit
tags:
  - pytorch
  - gan
  - dcgan
  - image-generation
  - celeba
  - educational
  - distributed-training
datasets:
  - celeba
pipeline_tag: image-to-image
---

# GANNs with Friends - Distributed DCGAN for CelebA

An educational distributed deep learning system where students participate as workers in a compute cluster to train a GAN (Generative Adversarial Network) to generate celebrity faces.

## Model description

This repository contains a **DCGAN (Deep Convolutional GAN)** trained on the CelebA dataset to generate 64x64 RGB images of human faces. The model follows the architecture described in the [original DCGAN paper](https://arxiv.org/abs/1511.06434).

### Architecture

**Generator:**
- Input: 100-dimensional latent noise vector
- 5 transposed convolutional layers with batch normalization and ReLU activation
- Output: 64x64x3 RGB image (Tanh activation, range [-1, 1])
- ~3.5M trainable parameters

**Discriminator:**
- Input: 64x64x3 RGB image
- 5 convolutional layers with batch normalization and LeakyReLU (0.2)
- Output: Single logit for real/fake classification
- ~2.7M trainable parameters

### Training configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 64x64 |
| Latent Dimension | 100 |
| Batch Size | 32-128 |
| Learning Rate | 0.0002 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Loss | BCEWithLogitsLoss |

## Dataset

This repository includes the **CelebA (CelebFaces Attributes)** dataset for training.

### About CelebA

- **Source**: Large-scale CelebFaces Attributes Dataset
- **Images**: ~200,000 celebrity face images
- **Resolution**: Original images center-cropped and resized to 64x64
- **Preprocessing**: Normalized to [-1, 1] range

### Dataset location

The dataset is stored in this repository at:
```
data/img_align_celeba.zip
```

### Dataset citation

```bibtex
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2015}
}
```

## Usage

### Loading the generator

```python
import torch
from huggingface_hub import hf_hub_download

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="gperdrizet/GANNs-with-friends",
    filename="checkpoint_latest.pth"
)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Define Generator architecture
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.main(z)

# Load weights
generator = Generator()
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Generate images
with torch.no_grad():
    noise = torch.randn(16, 100, 1, 1)
    fake_images = generator(noise)
    # Images are in range [-1, 1], normalize to [0, 1] for display
    fake_images = (fake_images + 1) / 2
```

### Downloading the dataset

```python
from huggingface_hub import hf_hub_download
import zipfile

# Download dataset
zip_path = hf_hub_download(
    repo_id="gperdrizet/GANNs-with-friends",
    filename="data/img_align_celeba.zip"
)

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('data/')
```

## Project: GANNs with Friends

This model is part of **GANNs with Friends**, an educational distributed deep learning project.

### Key features

- **Database-coordinated training**: Uses PostgreSQL as a coordination hub instead of complex networking
- **Fault tolerant**: Workers can disconnect/reconnect at any time
- **Flexible hardware**: CPU and GPU workers can participate together
- **Educational**: Teaches distributed systems, GANs, and parallel training simultaneously

### How it works

1. **Coordinator** creates work units (batches of image indices)
2. **Workers** (students' computers) poll the database for work
3. Workers compute gradients and upload them back
4. Coordinator aggregates gradients and updates the model
5. Model checkpoints are periodically pushed to this HuggingFace repository

### Learning outcomes

Students learn:
- **Distributed systems**: Coordination, fault tolerance, atomic operations
- **Deep learning**: GAN training, gradient aggregation, data parallelism
- **Practical skills**: PostgreSQL, PyTorch, collaborative computing

## Links

- **GitHub Repository**: [https://github.com/gperdrizet/GANNs-with-friends](https://github.com/gperdrizet/GANNs-with-friends)
- **Documentation**: [https://gperdrizet.github.io/GANNs-with-friends](https://gperdrizet.github.io/GANNs-with-friends)
- **Student Guide**: [How to participate as a worker](https://gperdrizet.github.io/GANNs-with-friends/guides/students.html)
- **Instructor Guide**: [Running the coordinator](https://gperdrizet.github.io/GANNs-with-friends/guides/instructors.html)

## Intended use

This model and dataset are intended for:
- **Educational purposes**: Learning about GANs, distributed training, and deep learning
- **Research**: Experimenting with GAN architectures and training techniques
- **Demonstrations**: Showcasing distributed machine learning concepts

## Limitations

- Generated images are 64x64 resolution
- Training quality depends on the number of participating workers and training duration
- The model may exhibit common GAN artifacts (mode collapse, training instability)
- Generated faces may not be perfectly realistic

## Ethical considerations

- This model generates synthetic faces and should not be used to create misleading content
- The CelebA dataset consists of celebrity images; generated images should not be used to impersonate real individuals
- Users should be transparent about AI-generated content

## Citation

If you use this project in your work, please cite:

```bibtex
@software{ganns_with_friends,
  title = {GANNs with Friends: Educational Distributed GAN Training},
  author = {Perdrizet, George},
  url = {https://github.com/gperdrizet/GANNs-with-friends},
  year = {2025}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/gperdrizet/GANNs-with-friends/blob/main/LICENSE) file for details.

## References

- [DCGAN Paper (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)
- [CelebA Dataset (Liu et al., 2015)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
