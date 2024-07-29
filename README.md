# Deep learning quantization with pytorch

This repository contains a project that aims to quantize a deep learning model. I've used specifically a VGG model. The code is inspired by the [MIT Han Lab](https://drive.google.com/file/d/124toPMHDd3z6LiXOhOgHPy6Wvb0Xzw3E/view) and is configured using Poetry for dependency management. The project utilizes torch-cpu for model operations. 


## Features
 * Int8 weights and activations quantization.
 * Per-channel weight quantization.
 * Affine quantization for activations.

## Performance
 * Float model accuracy : 92.95%
 * Int8 model accuracy  : 92.74%

## Installation

1. Clone the repository:

```
git clone https://github.com/AyoubMDL/torch_quantization.git
cd torch_quantization
```

2. Install dependencies
```
poetry install
```

3. Activate the virtual environment:

```
poetry shell
```

4. Run main file that loads pretrained VGG model and quantize it. Note that evaluation
on quantized model take more time as it uses Int operation on Conv and Linear layers

```
python main.py
```
