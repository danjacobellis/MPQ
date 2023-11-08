# Machine Perceptual Quality Evaluation Repository

This repository contains the code and experimental setup for evaluating the impact of severe lossy compression on audio and image models. The study investigates various perception models—including image classification, image segmentation, speech recognition, and music source separation—under severe lossy compression using several popular codecs.

## Datasets and Models

Below is a list of datasets and models used in our experiments, along with their corresponding tasks and evaluation metrics.

### Image classification
- **Model**: Vision Transformer ([`vit-base-patch16`](https://huggingface.co/google/vit-base-patch16-224))
  - **Pre-training**: ImageNet-21k
- **Evaluation dataset**: [ImageNet-1k](https://huggingface.co/datasets/imagenet-1k)
  - **Original compression**: JPEG at near-lossless quality settings. For additional details, see [Compression in the ImageNet Dataset](https://towardsdatascience.com/compression-in-the-imagenet-dataset-34c56d14d463) by Prof. Max Ehrlich
- **Evaluation metric**: Top-1 Accuracy

### Pneumonia classification
- **Model**: Vision Transformer ([`vit-xray-pneumonia-classification`](https://huggingface.co/lxyuan/vit-xray-pneumonia-classification))
  - **Pre-training**: ImageNet-21k
- **Evaluation dataset**: [Chest X-Ray](https://huggingface.co/datasets/keremberke/chest-xray-classification)
  - **Original compression**: Lossless
- **Evaluation metric**: Top-1 Accuracy

### Bean Disease Classification

- **Model**: Vision Transformer ([`vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans))
  - **Pre-training**: ImageNet-21k
- **Evaluation dataset**: [ibean:Bean disease dataset](https://github.com/AI-Lab-Makerere/ibean)
  - **Original compression**: Lossless
- **Evaluation metric**: Top-1 Accuracy

### Image Segmentation
- **Model**: SegFormer ([`segformer-b0-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512))
  - **Pre-training**: ImageNet-1k
- **Evaluation dataset**: [ADE20k](https://huggingface.co/datasets/scene_parse_150)
  - **Original compression**: JPEG at near-lossless quality settings.
- **Evaluation Metric**: Mean Intersection Over Union (MIOU)

### Speech Recognition
- **Model**: Whisper ([`whisper-medium`](https://huggingface.co/openai/whisper-medium))
  - **Pre-training** Proprietary dataset consisting of 680,000 hours of labeled audio.
- **Evaluation dataset**: [Common Voice 11.0](https://commonvoice.mozilla.org/en/datasets)
  - **Original compression**: MPEG Layer III (MP3) at near-lossless bitrates
- **Evaluation Metric**: Word Recognition Accuracy (WRA)

### Music Source Separation
- **Model**: [Demucs v3](https://github.com/facebookresearch/demucs/tree/v3)
- **Evaluation dataset**: [MUSDB18-HQ - an uncompressed version of MUSDB18](https://doi.org/10.5281/zenodo.3338373)
  - **Original compression**: Lossless
- **Evaluation Metric**: Signal-to-Distortion Ratio (SDR)


## Usage

To replicate the experiments or to evaluate your own models with the provided compression methods, please refer to the individual dataset and model links for setup and usage instructions.

## Contributing

We welcome contributions to this repository, whether it's in the form of additional datasets, models, compression methods, or improvements to the code. Please submit a pull request or open an issue to discuss your proposed changes.

## Citation

If you use the resources provided in this repository for your research, please cite the relevant papers linked with each dataset and model.
