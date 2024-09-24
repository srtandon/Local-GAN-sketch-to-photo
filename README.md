# Sketch to Photo GAN Model

This repository contains a Generative Adversarial Network (GAN) model designed to transform sketches into photorealistic images. The model is implemented using TensorFlow and Keras, based on the Wasserstein GAN architecture, and is designed to run locally on a single GPU. As you can see from the below example, this model is far from the level of popular generator models based on Stable Diffusion, Midjourney, etc. The purpose of this exercise was to learn about using GPUs and the basics of GAN models.

### Model Results
Test Photo:

![Test Photo:](https://github.com/srtandon/Local-GAN-sketch-to-photo/blob/main/outputs/test_input_01.png)

Output:

![Output:](https://github.com/srtandon/Local-GAN-sketch-to-photo/blob/main/outputs/test_output_001.png)

## Table of Contents

[Overview](#overview)

[Requirements](#requirements)

[Setup](#setup)

[Dataset](#dataset)

[Model Architecture](#model-architecture)

[Training](#training)

[Usage](#usage)

[Conclusion](#conclusion-and-limitations)

## Overview

This GAN model consists of a generator that transforms input sketches into photorealistic images and a discriminator that tries to distinguish between real photos and generated images. The model uses a combination of Wasserstein loss with gradient penalty and perceptual loss to achieve high-quality results. It is based on the Wasserstein GAN architecture, which provides improved stability during training compared to traditional GANs.

## Requirements

Python 3.10

TensorFlow 2.11

NumPy

Matplotlib

glob

A CUDA-capable GPU with at least 4GB of VRAM

CUDA 12.3

## Setup

1. Clone this repository:
```python
git clone <repository-url>
cd <repository-name>
```
2. Install the required packages:

   For Windows, you will need WSL2 to run TensorFlow with a GPU. Follow the instructions found [here](https://www.tensorflow.org/install).

3. Ensure you have a dataset to work with.
   
   This repo uses a dataset obtained from an e-learning project. Dataset not provided.
   
4. Make sure you have the latest NVIDIA GPU drivers and CUDA toolkit installed for optimal performance.
   
   You can use the command `nvidia-smi` to check the CUDA version and other details.

## Dataset

The dataset used for this model was pairs of photos and sketches.
Ensure your dataset is organized and update any paths in the code if necessary.

## Model Architecture
The model architecture is based on the Wasserstein GAN with gradient penalty (WGAN-GP), which provides more stable training and avoids mode collapse issues often seen in traditional GANs.

### Generator

Uses an encoder-decoder architecture with residual blocks
Input: 128x128x3 sketch image
Output: 128x128x3 photorealistic image
Features multiple convolutional and transposed convolutional layers
Utilizes batch normalization and LeakyReLU activations

### Discriminator

Convolutional network that outputs a single value
Input: 128x128x3 image (real or generated)
Output: Scalar indicating real/fake probability
Uses spectral normalization for improved stability

The model also incorporates a perceptual loss using a pre-trained VGG19 network to enhance the quality of generated images.

## Training
To train the model:

Ensure your dataset is correctly set up.
Verify that you have a CUDA-capable GPU with sufficient VRAM.

The training process will:

Run for 1000 epochs (configurable)
Print loss values for each epoch
Generate and save sample images every 10 epochs

Note: Training this model can be computationally intensive and may take several hours or days depending on your GPU capabilities and dataset size.

## Usage
After training, you can use the model to generate photos from sketches:
```python
# Preprocess photo to match training method
test_path = '/path/to/your/test/sketch.jpg'
test_image = load_img(test_path, target_size=(128, 128, 3))
test_image = img_to_array(test_image)
test_image = (test_image.astype(np.float32) - 127.5) / 127.5
zeros_array = np.zeros((1, 128, 128, 3))
zeros_array[0] = test_image.astype(np.float32)

# Generate result
result_photo = gen.predict(zeros_array)

# Show Image
plt.imshow(result_photo[0])
plt.show()
```

## Conclusion and Limitations

While this GAN model provides a solid foundation for transforming sketches into photorealistic images, it's important to acknowledge its limitations and potential areas for improvement:

### Image Size Limitations
1. Fixed Input/Output Size: The model is designed to work with 128x128 pixel images. This relatively small size may limit the level of detail in both input sketches and output photos.
2. Scalability Issues: Increasing the image size would require significant changes to the model architecture and could dramatically increase computational requirements and training time.

### Wasserstein Loss Considerations
1. Complexity: While Wasserstein loss can provide more stable training, it adds complexity to the model and may be more challenging to tune compared to traditional GAN losses.
2. Computational Overhead: The gradient penalty used in WGAN-GP adds computational overhead during training, which can slow down the process.
3. Potential Over-regularization: In some cases, the gradient penalty might over-regularize the discriminator, potentially affecting the quality of generated images.

### Other Limitations
1. Dataset Dependency: The model's performance is heavily dependent on the quality and diversity of the training dataset. A limited or biased dataset could result in poor generalization.
2. Single GPU Requirement: While designed for a single GPU, this might limit accessibility for users without appropriate hardware.
3. Limited Contextual Understanding: The model may struggle with complex sketches or generating context-dependent details that require higher-level understanding.
4. Lack of User Control: The current implementation doesn't allow for user input to guide specific aspects of the photo generation process.

### Potential Improvements
1. Implement progressive growing techniques to handle larger image sizes.
2. Explore other loss functions or hybrid approaches that might better balance stability and image quality.
3. Incorporate attention mechanisms to handle complex sketches better and improve detail generation.
4. Implement style transfer techniques to allow users to influence the style of the generated photos.
5. Optimize the code for multi-GPU training to improve accessibility and reduce training time.
6. Use a better dataset with human-annotated images.

Despite these limitations, this model serves as a valuable starting point for sketch-to-photo generation tasks. Future iterations could address these issues to create a more robust and versatile tool for image generation.

---

Note: This README assumes the code is part of a larger project structure. You may need to adjust file paths and import statements depending on your specific project setup. Additionally, while the model is designed to run on a single GPU, performance may vary depending on your specific hardware configuration.
