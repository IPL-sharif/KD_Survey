## 🚀 [TMLR'25] A Comprehensive Survey on Knowledge Distillation
[![arXiv](https://img.shields.io/badge/arXiv-2503.12067-<COLOR>.svg)](https://arxiv.org/abs/2503.12067)

This repository belongs to the paper: [A Comprehensive Survey on Knowledge Distillation](https://openreview.net/forum?id=3cbJzdR78B).

## 📝 Abstract
Deep Neural Networks (DNNs) have achieved notable performance in the fields of computer vision and natural language processing with various applications in both academia and industry. However, with recent advancements in DNNs and transformer models with a tremendous number of parameters, deploying these large models on edge devices causes serious issues such as high runtime and memory consumption. This is especially concerning with the recent large-scale foundation models, Vision-Language Models (VLMs), and Large Language Models (LLMs). Knowledge Distillation (KD) is one of the prominent techniques proposed to address the aforementioned problems using a teacher-student architecture. More specifically, a lightweight student model is trained using additional knowledge from a cumbersome teacher model. In this work, a comprehensive survey of knowledge distillation methods is proposed. This includes reviewing KD from different aspects: distillation sources, distillation schemes, distillation algorithms, distillation by modalities, applications of distillation, and comparison among existing methods. In contrast to most existing surveys, which are either outdated or simply update former surveys, this work proposes a comprehensive survey with a new point of view and representation structure  that categorizes and investigates the most recent methods in knowledge distillation. This survey considers various critically important subcategories, including KD for diffusion models, 3D inputs, foundational models, transformers, and LLMs. Furthermore, existing challenges in KD and possible future research directions are discussed.

<p align="center">
 <img src="https://raw.githubusercontent.com/IPL-sharif/KD_Survey/refs/heads/main/Figures/github_diagram.png"  style="max-width: 100%; height: auto;"/>
</p>


## 📚 Existing Surveys on Knowledge Distillation
- **Knowledge Distillation: A Survey**, IJCV 2021, [ :link: ](https://arxiv.org/abs/2006.05525)
- **Knowledge Distillation in Deep Learning and its Applications**, Peer J Computer Science 2021, [ :link: ](https://arxiv.org/abs/2007.09029)
- **Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks**, T-PAMI 2021, [ :link: ](https://arxiv.org/abs/2004.05937)
- **Teacher-Student Architecture for Knowledge Distillation: A Survey**, arXiv 2023, [ :link: ](https://arxiv.org/abs/2308.04268)
- **Categories of Response-Based, Feature-Based, and Relation-Based Knowledge Distillation**, arXiv 2023, [ :link: ](https://arxiv.org/abs/2306.10687)
- **A survey on knowledge distillation: Recent advancements**, Machine Learning with Applications 2024, [ :link: ](https://www.sciencedirect.com/science/article/pii/S2666827024000811)


## 📑 Contents

### [Distillation Sources](Sources/)
- [Logit-based Distillation](Sources/README.md#Logit-based-Distillation)
- [Feature-based Distillation](Sources/README.md#Feature-based-Distillation)
- [Similarity-based Distillation](Sources/README.md#Similarity-based-Distillation)

### [Distillation Schemes](Schemes/)
- [Offline Distillation](Schemes/README.md#Offline-Distillation)
- [Online Distillation](Schemes/README.md#Online-Distillation)
- [Self-Distillation](Schemes/README.md#Self-Distillation)

### [Distillation Algorithms](Algorithms/)
- [Attention Distillation](Algorithms/README.md#Attention-Distillation)
- [Adversarial Distillation](Algorithms/README.md#Adversarial-Distillation)
- [Multi-teacher Distillation](Algorithms/README.md#Multi-teacher-Distillation)
- [Cross-modal Distillation](Algorithms/README.md#Cross-modal-Distillation)
- [Graph-based Distillation](Algorithms/README.md#Graph-based-Distillation)
- [Adaptive Distillation](Algorithms/README.md#Adaptive-Distillation)
- [Contrastive Distillation](Algorithms/README.md#Contrastive-Distillation)

### [Distillation by Modolities](Modalities/)
- [3D Input](Modalities/README.md#3d-input)
- [Multi-veiw](Modalities/README.md#multi-view)
- [Text](Modalities/README.md#text)
- [Speech](Modalities/README.md#speech)
- [Video](Modalities/README.md#video)


### [Applications of Distillation](Applications/)
- [Large Language Models](Applications/README.md#Self-Supervised-Learning)
- [Foundation Models](Applications/README.md#foundation-models)
- [Self-Supervised Learning](Applications/README.md#Large-Language-Models)
- [Diffusion Models](Applications/README.md#diffusion-models)
- [Visual Recognition](Applications/README.md#knowledge-distillation-in-visual-recognition)

## 📜 Citation
If you use this repository for your research or wish to refer to our distillation methods, please use the following BibTeX entries:
```bibtex
@article{mansourian2025a,
title={A Comprehensive Survey on Knowledge Distillation},
author={Amir M. Mansourian and Rozhan Ahmadi and Masoud Ghafouri and Amir Mohammad Babaei and Elaheh Badali Golezani and Zeynab yasamani ghamchi and Vida Ramezanian and Alireza Taherian and Kimia Dinashi and Amirali Miri and Shohreh Kasaei},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025}
}

@article{mansourian2025enriching,
  title={Enriching Knowledge Distillation with Cross-Modal Teacher Fusion},
  author={Mansourian, Amir M and Babaei, Amir Mohammad and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2511.09286},
  year={2025}
}

@article{mansourian2025aicsd,
  title={AICSD: Adaptive inter-class similarity distillation for semantic segmentation},
  author={Mansourian, Amir M and Ahmadi, Rozhan and Kasaei, Shohreh},
  journal={Multimedia Tools and Applications},
  pages={1--20},
  year={2025},
  publisher={Springer}
}

@article{mansourian2024attention,
  title={Attention-guided feature distillation for semantic segmentation},
  author={Mansourian, Amir M and Jalali, Arya and Ahmadi, Rozhan and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2403.05451},
  year={2024}
}

@inproceedings{ahmadi2024leveraging,
  title={Leveraging swin transformer for local-to-global weakly supervised semantic segmentation},
  author={Ahmadi, Rozhan and Kasaei, Shohreh},
  booktitle={2024 13th Iranian/3rd International Machine Vision and Image Processing Conference (MVIP)},
  pages={1--7},
  year={2024},
  organization={IEEE}
}

@article{mansourian2023efficient,
  title={An Efficient Knowledge Distillation Architecture for Real-time Semantic Segmentation},
  author={Mansourian, Amir M and Karimi Bavandpour, Nader and Kasaei, Shohreh},
  journal={AUT Journal of Modeling and Simulation},
  volume={55},
  number={1},
  pages={99--108},
  year={2023}
}

@article{ghorbani2020your,
  title={Be Your Own Best Competitor! Multi-Branched Adversarial Knowledge Transfer},
  author={Ghorbani, Mahdi and Fooladgar, Fahimeh and Kasaei, Shohreh},
  journal={arXiv preprint arXiv:2010.04516},
  year={2020}
}

@inproceedings{bavandpour2020class,
  title={Class attention map distillation for efficient semantic segmentation},
  author={Bavandpour, Nader Karimi and Kasaei, Shohreh},
  booktitle={2020 International Conference on Machine Vision and Image Processing (MVIP)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

```
