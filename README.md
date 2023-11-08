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

## Compression Methods

In our experiments, we employ a variety of compression methods to evaluate the robustness of machine perception models. Below is a detailed description of each method and its implementation:

### Image Compression

- **JPEG**: An older but still widely used DCT-based image compression method. We use the implementation in the [Pillow library](https://github.com/python-pillow/Pillow).

- **WEBP**: A modern image compression standard that also uses transform coding. We use the implementation available in the Pillow library.

- **Minnen, Ballé, Toderici 2018 (mbt2018)**: A distortion-optimized neural image compression model presented in the paper by Minnen et al., 2018. We use the implementation available in the [CompressAI library](https://github.com/InterDigitalInc/CompressAI).

- **HiFiC**: A generative neural image compression model optimized using an adversarial loss, as described by Mentzer et al., 2020. We use the implementation available in the [Tensorflow Compression library](https://github.com/tensorflow/compression).

### Audio Compression

- **MPEG Layer III (MP3)**: The most widely used lossy audio compression standard, which employs transform coding. We use the implementation available in the [torchaudio library](https://github.com/pytorch/audio).

- **Opus**: A versatile audio codec suitable for both general audio and speech. We use the implementation available in the [opuspy library](https://github.com/elevenlabs/opuspy).

- **EnCodec**: A neural audio compression model that uses an adversarial loss, as proposed by Défossez et al., 2022. We use the official implementation available on [GitHub](https://github.com/facebookresearch/encodec).

## Deep Similarity Metrics

Our study also incorporates advanced deep similarity metrics to assess the perceptual quality of images and audio after compression. These metrics go beyond traditional measures like PSNR or SSIM, capturing more nuanced perceptual differences.

### Image Similarity

- **Learned Perceptual Image Patch Similarity (LPIPS)**: The implementation is available in the [Pytorch Image Quality (piq) library](https://github.com/photosynthesis-team/piq).

### Audio Similarity

- **Contrastive Deep Perceptual Audio Similarity Metric (CDPAM)**: The implementation is available in the [Official repository](https://github.com/pranaymanocha/PerceptualAudio).


## Usage

To replicate the experiments or to evaluate your own models with the provided compression methods, please refer to the individual dataset and model links for setup and usage instructions, then run the notebook corresponding to the desired experiment.

## Contributing

We welcome contributions to this repository, whether it's in the form of additional datasets, models, compression methods, or improvements to the code. Please submit a pull request or open an issue to discuss your proposed changes.

## Citations

```bibtex
@article{minnen2018joint,
  title={Joint autoregressive and hierarchical priors for learned image compression},
  author={Minnen, David and Ball{\'e}, Johannes and Toderici, George D},
  journal={Advances in neural information processing systems},
  volume={31},
  year={2018}
}
@article{begaint2020compressai,
  title={Compressai: a pytorch library and evaluation platform for end-to-end compression research},
  author={B{\'e}gaint, Jean and Racap{\'e}, Fabien and Feltman, Simon and Pushparaja, Akshay},
  journal={arXiv preprint arXiv:2011.03029},
  year={2020}
}
@article{mentzer2020high,
  title={High-fidelity generative image compression},
  author={Mentzer, Fabian and Toderici, George D and Tschannen, Michael and Agustsson, Eirikur},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={11913--11924},
  year={2020}
}
@article{balle2022tensorflow,
  title={TensorFlow compression: Learned data compression},
  author={Ball{\'e}, Johannes and Hwang, Sung Jin and Agustsson, Eirikur},
  journal={TensorFlow Compression: Learned data compression},
  year={2022}
}
@article{defossez2022high,
  title={High fidelity neural audio compression},
  author={D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}

@inproceedings{wang2017chestx,
  title={Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2097--2106},
  year={2017}
}

@article{singh2023classification,
  title={Classification of Beans Leaf Diseases using Fine Tuned CNN Model},
  author={Singh, Vimal and Chug, Anuradha and Singh, Amit Prakash},
  journal={Procedia Computer Science},
  volume={218},
  pages={348--356},
  year={2023},
  publisher={Elsevier}
}

@inproceedings{zhou2017scene,
  title={Scene parsing through ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={633--641},
  year={2017}
}

@article{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={12077--12090},
  year={2021}
}

@inproceedings{ardila2020common,
  title={Common Voice: A Massively-Multilingual Speech Corpus},
  author={Ardila, Rosana and Branson, Megan and Davis, Kelly and Kohler, Michael and Meyer, Josh and Henretty, Michael and Morais, Reuben and Saunders, Lindsay and Tyers, Francis and Weber, Gregor},
  booktitle={Proceedings of the Twelfth Language Resources and Evaluation Conference},
  pages={4218--4222},
  year={2020}
}

@inproceedings{radford2023robust,
  title={Robust speech recognition via large-scale weak supervision},
  author={Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  booktitle={International Conference on Machine Learning},
  pages={28492--28518},
  year={2023},
  organization={PMLR}
}

@article{rafii2017musdb18,
  title={MUSDB18-a corpus for music separation},
  author={Rafii, Zafar and Liutkus, Antoine and St{\"o}ter, Fabian-Robert and Mimilakis, Stylianos Ioannis and Bittner, Rachel},
  year={2017}
}

@article{defossez2021hybrid,
  title={Hybrid spectrogram and waveform source separation},
  author={D{\'e}fossez, Alexandre},
  journal={arXiv preprint arXiv:2111.03600},
  year={2021}
}

@inproceedings{yang2022torchaudio,
  title={Torchaudio: Building blocks for audio and speech processing},
  author={Yang, Yao-Yuan and Hira, Moto and Ni, Zhaoheng and Astafurov, Artyom and Chen, Caroline and Puhrsch, Christian and Pollack, David and Genzel, Dmitriy and Greenberg, Donny and Yang, Edward Z and others},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6982--6986},
  year={2022},
  organization={IEEE}
}

@inproceedings{zhang2018unreasonable,
  title={The unreasonable effectiveness of deep features as a perceptual metric},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A and Shechtman, Eli and Wang, Oliver},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={586--595},
  year={2018}
}
@misc{kastryulin2022piq,
  title = {PyTorch Image Quality: Metrics for Image Quality Assessment},
  url = {https://arxiv.org/abs/2208.14818},
  author = {Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V.},
  doi = {10.48550/ARXIV.2208.14818},
  publisher = {arXiv},
  year = {2022}
}
@misc{piq,
  title={{PyTorch Image Quality}: Metrics and Measure for Image Quality Assessment},
  url={https://github.com/photosynthesis-team/piq},
  note={Open-source software available at https://github.com/photosynthesis-team/piq},
  author={Sergey Kastryulin and Dzhamil Zakirov and Denis Prokopenko},
  year={2019}
}
@inproceedings{manocha2021cdpam,
  title={CDPAM: Contrastive learning for perceptual audio similarity},
  author={Manocha, Pranay and Jin, Zeyu and Zhang, Richard and Finkelstein, Adam},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={196--200},
  year={2021},
  organization={IEEE}
}
```
