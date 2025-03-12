# Modalities

* [3D Input](#3d-input)
  *  [Object Detection](#Object-Detection)
  *  [Shape Classification](#Shape-Classification)
  *  [Semantic Segmentation](#Semantic-Segmentation)
  *  [Domain Adaptation](#Domain-Adaptation)
  *  [Depth Estimation](#Depth-Estimation)
  *  [Representation Learning](#Representation)
  *  [Recognition](#Recognition)
  *  [Completion](#Completion)
  *  [Registartion](#Registartion)
  *  [Other 3D Tasks](#Other-3D-Tasks)
 
* [Multi-View](#multi-view)
  *  [3D Object Detection](#3d-object-detection)
  *  [3D Shape Recognition](#3d-shape-recognition)
  *  [Other Tasks](#other-tasks)


* [Text](#text)
  * [Neural Machine Translation (NMT)](#Neural-Machine-Translation-NMT)
  * [Question Answering (QA)](#Question-Answering-QA)
  * [Text Generation](#Text-Generation)
  * [Event Detection](#Event-Detection)
  * [Document Retrieval](#Document-Retrieval)
  * [Text Recognition](#Text-Recognition)
  * [Named Entity Recognition (NER)](#Named-Entity-Recognition-NER)
  * [Text Summarization](#Text-Summarization)
  * [Natural Language Understanding (NLU)](#Natural-Language-Understanding-NLU)
  * [Sentiment Analysis](#Sentiment-Analysis)
  * [Text Classification](#Text-Classification) 

* [Speech](#speech)
  * [Speech Recognition (ASR)](#Speech-Recognition-ASR)
  * [Speech Enhancement](#Speech-Enhancement)
  * [Speaker Recognition and Verification](#Speaker-Recognition-and-Verification)
  * [Speech Translation](#Speech-Translation)
  * [Speech Synthesis (Text-to-Speech)](#Speech-Synthesis-Text-to-Speech)
  * [Speech Separation](#Speech-Separation)
  * [Spoken Language Identification and Understanding](#Spoken-Language-Identification-and-Understanding)
  * [Deepfake Speech and Spoofing Detection](#Deepfake-Speech-and-Spoofing-Detection)
  * [Audio Classification and Tagging](#Audio-Classification-and-Tagging)
  * [Spoken Question Answering and Conversational AI](#Spoken-Question-Answering-and-Conversational-AI)
  * [Audio Captioning and Retrieval](#Audio-Captioning-and-Retrieval) 

* [Video](#video)
  
---
## 3D Input
### Object Detection
* **Weak-to-strong 3d object detection with x-ray distillation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Gambashidze_Weak-to-Strong_3D_Object_Detection_with_X-Ray_Distillation_CVPR_2024_paper.html) [ :octocat: ](https://github.com/sakharok13/X-Ray-Teacher-Patching-Tools)
* **RadarDistill: Boosting Radar-based Object Detection Performance via Knowledge Distillation from LiDAR Features**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Bang_RadarDistill_Boosting_Radar-based_Object_Detection_Performance_via_Knowledge_Distillation_from_CVPR_2024_paper.html)
* **CRKD: Enhanced Camera-Radar Object Detection with Cross-modality Knowledge Distillation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_CRKD_Enhanced_Camera-Radar_Object_Detection_with_Cross-modality_Knowledge_Distillation_CVPR_2024_paper.html) [ :octocat: ](https://github.com/Song-Jingyu/CRKD)
* **Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection**, AAAI 2024, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/28016) [ :octocat: ](https://github.com/ylwhxht/SRKD-DRET)
* **PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_PointDistiller_Structured_Knowledge_Distillation_Towards_Efficient_and_Compact_3D_Detection_CVPR_2023_paper.html) [ :octocat: ](https://github.com/RunpeiDong/PointDistiller)
* **itKD: Interchange Transfer-Based Knowledge Distillation for 3D Object Detection**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Cho_itKD_Interchange_Transfer-Based_Knowledge_Distillation_for_3D_Object_Detection_CVPR_2023_paper.html) [ :octocat: ](https://github.com/hyeon-jo/interchange-transfer-KD)
* **Distilling Focal Knowledge From Imperfect Expert for 3D Object Detection**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Zeng_Distilling_Focal_Knowledge_From_Imperfect_Expert_for_3D_Object_Detection_CVPR_2023_paper.html) [ :octocat: ](https://github.com/OpenDriveLab/Birds-eye-view-Perception)
* **X3KD: Knowledge Distillation Across Modalities, Tasks and Stages for Multi-Camera 3D Object Detection**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Klingner_X3KD_Knowledge_Distillation_Across_Modalities_Tasks_and_Stages_for_Multi-Camera_CVPR_2023_paper.html)
* **UniDistill: A Universal Cross-Modality Knowledge Distillation Framework for 3D Object Detection in Bird's-Eye View**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_UniDistill_A_Universal_Cross-Modality_Knowledge_Distillation_Framework_for_3D_Object_CVPR_2023_paper.html) [ :octocat: ](https://github.com/megvii-research/CVPR2023-UniDistill)
* **Representation Disparity-aware Distillation for 3D Object Detection**, ICCV 2023, [ :link: ](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Representation_Disparity-aware_Distillation_for_3D_Object_Detection_ICCV_2023_paper.html)
* **DistillBEV: Boosting Multi-Camera 3D Object Detection with Cross-Modal Knowledge Distillation**, ICCV 2023, [ :link: ](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DistillBEV_Boosting_Multi-Camera_3D_Object_Detection_with_Cross-Modal_Knowledge_Distillation_ICCV_2023_paper.html) [ :octocat: ](https://github.com/qcraftai/distill-bev)
* **Voxel-to-Pillar: Knowledge Distillation of 3D Object Detection in Point Cloud**, ACM 2023, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3651640.3651652)
* **LiDAR Distillation: Bridging the Beam-Induced Domain Gap for 3D Object Detection**, ECCV 2022, [ :link: ](https://arxiv.org/pdf/2203.14956) [ :octocat: ](https://github.com/weiyithu/LiDAR-Distillation)
* **Cross-Modality Knowledge Distillation Network for Monocular 3D Object Detection**, ECCV 2022, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-031-20080-9_6) [ :octocat: ](https://github.com/Cc-Hy/CMKD)
* **Towards Efficient 3D Object Detection with Knowledge Distillation**, NeurIPS 2022, [ :link: ](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8625a8c2be8ba5197b7a14833dbea8ac-Abstract-Conference.html) [ :octocat: ](https://github.com/CVMI-Lab/SparseKD)

### Shape Classification
* **Feature Adversarial Distillation for Point Cloud Classification**, ICIP 2023, [ :link: ](https://arxiv.org/abs/2306.14221)
* **Joint graph entropy knowledge distillation for point cloud classification and robustness against corruptions**, Information Sciences 2023, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S0020025523011271)
* **Efficient Point Cloud Classification via Offline Distillation Framework and Negative-Weight Self-Distillation Technique**, arXiv 2024, [ :link: ](https://arxiv.org/abs/2409.02020)
* **Cascaded Network with Hierarchical Self-Distillation for Sparse Point Cloud Classification**, ICME 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10687949/) [ :octocat: ](https://github.com/ky-zhou/pointhsd)

### Semantic Segmentation
* **Learning 3D Semantic Segmentation with only 2D Image Supervision**, IC3DV 2021,  [ :link: ](https://arxiv.org/abs/2110.11325)
* **3D-to-2D Distillation for Indoor Scene Parsing**, CVPR 2021,  [ :link: ](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_3D-to-2D_Distillation_for_Indoor_Scene_Parsing_CVPR_2021_paper.html) [ :octocat: ](https://github.com/liuzhengzhe/3D-to-2D-Distillation-for-Indoor-Scene-Parsing)
* **Perturbed Self-Distillation: Weakly Supervised Large-Scale Point Cloud Semantic Segmentation**, ICCV 2021,  [ :link: ](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Perturbed_Self-Distillation_Weakly_Supervised_Large-Scale_Point_Cloud_Semantic_Segmentation_ICCV_2021_paper.html) [ :octocat: ](https://github.com/Yachao-Zhang/PSD)
* **Point-to-voxel knowledge distillation for lidar semantic segmentation**, CVPR 2022,  [ :link: ](https://openaccess.thecvf.com/content/CVPR2022/html/Hou_Point-to-Voxel_Knowledge_Distillation_for_LiDAR_Semantic_Segmentation_CVPR_2022_paper.html) [ :octocat: ](https://github.com/cardwing/Codes-for-PVKD)
* **Multi-to-Single Knowledge Distillation for Point Cloud Semantic Segmentation**, IEEE International Conference on Robotics and Automation (ICRA) 2023,  [ :link: ](https://arxiv.org/abs/2304.14800) [ :octocat: ](https://github.com/skyshoumeng/M2SKD)
* **Label-Guided Knowledge Distillation for Continual Semantic Segmentation on 2D Images and 3D Point Clouds**, ICCV 2023,  [ :link: ](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Label-Guided_Knowledge_Distillation_for_Continual_Semantic_Segmentation_on_2D_Images_ICCV_2023_paper.html) [ :octocat: ](https://github.com/Ze-Yang/LGKD)
* **Smaller3D: Smaller models for 3D semantic segmentation using Minkowski engine and knowledge distillation methods**, arXiv  2023,  [ :link: ](https://arxiv.org/abs/2305.03188#:~:text=4%20May%202023%5D-,Smaller3d%3A%20Smaller%20Models%20for%203D%20Semantic%20Segmentation%20Using,Engine%20and%20Knowledge%20Distillation%20Methods&text=There%20are%20various%20optimization%20techniques,how%20do%20calculate%20in%203D.) [ :octocat: ](https://github.com/madanela/smaller3d)
* **CMDFusion: Bidirectional Fusion Network with Cross-modality Knowledge Distillation for LIDAR Semantic Segmentation**, IEEE Robotics and Automation Letters 2023,  [ :link: ](https://arxiv.org/abs/2307.04091) [ :octocat: ](https://github.com/Jun-CEN/CMDFusion)
* **Knowledge Distillation from 3D to Bird's-Eye-View for LiDAR Semantic Segmentation**, IEEE International Conference on Multimedia and Expo (ICME) 2023,  [ :link: ](https://arxiv.org/abs/2304.11393) [ :octocat: ](https://github.com/fengjiang5/Knowledge-Distillation-from-Cylinder3D-to-PolarNet)
* **PartDistill: 3D Shape Part Segmentation by Vision-Language Model Distillation**, CVPR 2024,  [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Umam_PartDistill_3D_Shape_Part_Segmentation_by_Vision-Language_Model_Distillation_CVPR_2024_paper.html) [ :octocat: ](https://github.com/ardianumam/PartDistill)
* **Segment Any Point Cloud Sequences by Distilling Vision Foundation Models**, NeurIPS 2024,  [ :link: ](https://arxiv.org/abs/2306.09347)  [ :octocat: ](https://github.com/youquanl/Segment-Any-Point-Cloud)

### Domain Adaptation
* **Self-Distillation for Unsupervised 3D Domain Adaptation**, WACV 2023, [ :link: ](https://openaccess.thecvf.com/content/WACV2023/html/Cardace_Self-Distillation_for_Unsupervised_3D_Domain_Adaptation_WACV_2023_paper.html) [ :octocat: ](https://github.com/CVLAB-Unibo/Feature-Distillation-for-3D-UDA/tree/main)
* **Cross-modal unsupervised domain adaptation for 3d semantic segmentation via bidirectional fusion-then-distillation**, ACM 2023,  [ :link: ](https://dl.acm.org/doi/10.1145/3581783.3612013)
* **Self-ensembling for 3D point cloud domain adaptation**, Image and Vision Computing 2024, [ :link: ](https://arxiv.org/abs/2112.05301)

### Depth Estimation
* **MVP-Net: Multi-View Depth Image Guided Cross-Modal Distillation Network for Point Cloud Upsampling**, ACM 2024, [ :link: ](https://dl.acm.org/doi/10.1145/3664647.3681562)
* **Improving Accuracy and Efficiency of Monocular Depth Estimation in Power Grid Environments Using Point Cloud Optimization and Knowledge Distillation**, Energies 2024, [ :link: ](https://www.mdpi.com/1996-1073/17/16/4068) 
* **LiRCDepth: Lightweight Radar-Camera Depth Estimation via Knowledge Distillation and Uncertainty Guidance**, arXiv 2024, [ :link: ](https://www.arxiv.org/abs/2412.16380) [ :octocat: ](https://github.com/harborsarah/LiRCDepth)

### Representation Learning
* **Learning from 2D: Contrastive Pixel-to-Point Knowledge Transfer for 3D Pretraining**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2104.04687)
* **Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data**, CVPR 2022, [ :link: ](https://openaccess.thecvf.com/content/CVPR2022/html/Sautier_Image-to-Lidar_Self-Supervised_Distillation_for_Autonomous_Driving_Data_CVPR_2022_paper.html) [ :octocat: ](https://github.com/valeoai/SLidR)
* **Distillation with Contrast is All You Need for Self-Supervised Point Cloud**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2202.04241)
* **Data Efficient 3D Learner via Knowledge Transferred from 2D Model**, ECCV 2022, [ :link: ](https://arxiv.org/abs/2203.08479) [ :octocat: ](https://github.com/bryanyu1997/Data-Efficient-3D-Learner)
* **Let Images Give You More: Point Cloud Cross-Modal Training for Shape Analysis**, NeurIPS 2022, [ :link: ](https://arxiv.org/abs/2210.04208) [ :octocat: ](https://github.com/ZhanHeshen/PointCMT)
* **3D Point Cloud Pre-Training with Knowledge Distilled from 2D Images**, ICME 2022, [ :link: ](https://arxiv.org/abs/2212.08974)
* **LiDAR2Map: In Defense of LiDAR-Based Semantic Map Construction Using Online Camera Distillation**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_LiDAR2Map_In_Defense_of_LiDAR-Based_Semantic_Map_Construction_Using_Online_CVPR_2023_paper.html) [ :octocat: ](https://github.com/songw-zju/LiDAR2Map)
* **Learning 3D Representations from 2D Pre-trained Models via Image-to-Point Masked Autoencoders**, CVPR 2023, [ :link: ](https://arxiv.org/pdf/2212.06785) [ :octocat: ](https://github.com/ZrrSkywalker/I2P-MAE)
* **Complete-to-Partial 4D Distillation for Self-Supervised Point Cloud Sequence Representation Learning**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Complete-to-Partial_4D_Distillation_for_Self-Supervised_Point_Cloud_Sequence_Representation_Learning_CVPR_2023_paper.html) [ :octocat: ](https://github.com/dongyh20/C2P)
* **Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining**, ICML 2023, [ :link: ](https://arxiv.org/abs/2302.02318) [ :octocat: ](https://github.com/qizekun/ReCon)
* **Multi-View Representation is What You Need for Point-Cloud Pre-Training**, ICLR 2024, [ :link: ](https://openreview.net/forum?id=imZcqOrbig)
* **HVDistill: Transferring Knowledge from Images to Point Clouds via Unsupervised Hybrid-View Distillation**, IJCV 2024, [ :link: ](https://arxiv.org/abs/2403.11817) [ :octocat: ](https://github.com/zhangsha1024/HVDistill)
* **Joint representation learning for text and 3d point cloud**, Pattern Recognition 2024, [ :link: ](https://arxiv.org/abs/2301.07584)
* **Diffusion 3d features (diff3f): Decorating untextured shapes with distilled semantic features**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Dutt_Diffusion_3D_Features_Diff3F_Decorating_Untextured_Shapes_with_Distilled_Semantic_CVPR_2024_paper.html) [ :octocat: ](https://diff3f.github.io/)

### Recognition
* **DistilVPR: Cross-Modal Knowledge Distillation for Visual Place Recognition**, AAAI 2024, [ :link: ](https://arxiv.org/abs/2312.10616) [ :octocat: ](https://github.com/sijieaaa/DistilVPR)
* **PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition**, IEEE Transactions on Multimedia 2023, [ :link: ](https://arxiv.org/abs/2207.03128) [ :octocat: ](https://github.com/keeganhk/PointMCD)

### Completion
#### Shape Completion
* **RaPD, Reconstruction-Aware Prior Distillation for Semi-supervised Point Cloud Completion**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2204.09186) 
* **Loss Distillation via Gradient Matching for Point Cloud Completion with Weighted Chamfer Distance**, EEE/RSJ IROS 2024, [ :link: ](https://arxiv.org/abs/2409.06171) [ :octocat: ](https://github.com/Zhang-VISLab/IROS2024-LossDistillationWeightedCD)
* **Enhancing the Encoding Process in Point Cloud Completion**, ICCPR 2024, [ :link: ](https://dl.acm.org/doi/10.1145/3704323.3704330)

#### Scene Completion
* **SCPNet: Semantic Scene Completion on Point Cloud**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Xia_SCPNet_Semantic_Scene_Completion_on_Point_Cloud_CVPR_2023_paper.html) 
* **Not All Voxels Are Equal: Hardness-Aware Semantic Scene Completion with Self-Distillation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Not_All_Voxels_Are_Equal_Hardness-Aware_Semantic_Scene_Completion_with_CVPR_2024_paper.html) [ :octocat: ](https://github.com/songw-zju/HASSC)
* **Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion**, arXive 2024, [ :link: ](https://arxiv.org/abs/2412.03515) [ :octocat: ](https://github.com/happyw1nd/ScoreLiDAR)
* **Voxel Proposal Network via Multi-Frame Knowledge Distillation for Semantic Scene Completion**, NeurIPS 2025, [ :link: ](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b734c30b9c955c535e333f0301f5e45c-Abstract-Conference.html) 

#### Depth Completion
* **LiDAR Depth Completion Using Color-Embedded Information via Knowledge Distillation**, IEEE Transactions on Intelligent Transportation Systems 2021, [ :link: ](https://ieeexplore.ieee.org/document/9635718)
* **Monitored Distillation for Positive Congruent Depth Completion**, ECCV 2020, [ :link: ](https://arxiv.org/abs/2203.16034) [ :octocat: ](https://github.com/alexklwong/mondi-python)
* **ADNet: Non-Local Affinity Distillation Network for Lightweight Depth Completion With Guidance From Missing LiDAR Points**, IEEE Robotics and Automation Letters 2024, [ :link: ](https://ieeexplore.ieee.org/document/10598333)
* **Multi-Scale Distillation for Low Scanline Resolution Depth Completion**, International Conference on Computer and Communication 2024, [ :link: ](https://ieeexplore.ieee.org/document/10603293)


### Registration
* **Unsupervised Point Cloud Registration with Self-Distillation**, arXiv 2024, [ :link: ](https://arxiv.org/abs/2409.07558) [ :octocat: ](https://github.com/boschresearch/direg)
* **Knowledge distillation-based point cloud registration method**, International Conference on Computer Graphics 2024, [ :link: ](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13105/3026745/Knowledge-distillation-based-point-cloud-registration-method/10.1117/12.3026745.short) 


### Other 3D Tasks
* **Diffusion Time-step Curriculum for One Image to 3D Generation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Diffusion_Time-step_Curriculum_for_One_Image_to_3D_Generation_CVPR_2024_paper.html) [ :octocat: ](https://github.com/yxymessi/DTC123)
* **3D Paintbrush: Local Stylization of 3D Shapes with Cascaded Score Distillation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2024/html/Decatur_3D_Paintbrush_Local_Stylization_of_3D_Shapes_with_Cascaded_Score_CVPR_2024_paper.html) [ :octocat: ](https://threedle.github.io/3d-paintbrush/)
* **Resolution-free Point Cloud Sampling Network with Data Distillation**, ECCV 2022, [ :link: ](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4326_ECCV_2022_paper.php) [ :octocat: ](https://github.com/Tianxinhuang/PCDNet)
* **Open-Vocabulary Affordance Detection using Knowledge Distillation and Text-Point Correlation**,  [ :link: ](https://arxiv.org/abs/2309.10932) [ :octocat: ](https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-using-Knowledge-Distillation-and-Text-Point-Correlation)



## Multi-View
### 3D Object Detection
* **Distillbev: Boosting multi-camera 3d object detection with cross-modal knowledge distillation**, ICCV 2023, [ :link: ](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DistillBEV_Boosting_Multi-Camera_3D_Object_Detection_with_Cross-Modal_Knowledge_Distillation_ICCV_2023_paper.html) [ :octocat: ](https://github.com/qcraftai/distill-bev)
* **X3kd: Knowledge distillation across modalities, tasks and stages for multi-camera 3d object detection**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Klingner_X3KD_Knowledge_Distillation_Across_Modalities_Tasks_and_Stages_for_Multi-Camera_CVPR_2023_paper.html)
* **Simdistill: Simulated multi-modal distillation for bev 3d object detection**, AAAI conference on artificial intelligence , [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/28577) [ :octocat: ](https://github.com/ViTAE-Transformer/SimDistill)
* **Fsd-bev: Foreground self-distillation for multi-view 3d object detection**, ECCV 2024, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-031-73242-3_7) [ :octocat: ](https://github.com/CocoBoom/fsd-bev)
* **Bev-lgkd: A unified lidar-guided knowledge distillation framework for bev 3d object detection**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2212.00623) [ :octocat: ](https://github.com/NorthSummer/LGKD)
* **Bevdistill: Cross-modal bev distillation for multi-view 3d object detection**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2211.09386) [ :octocat: ](https://github.com/zehuichen123/BEVDistill)
* **Stxd: structural and temporal cross-modal distillation for multi-view 3d object detection**, NeurIPS 2023 [ :link: ](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html)

### 3D Shape Recognition
* **PointMCD: Boosting Deep Point Cloud Encoders via Multi-view Cross-modal Distillation for 3D Shape Recognition**, IEEE Transactions on Multimedia 2023, [ :link: ](https://arxiv.org/abs/2207.03128) [ :octocat: ](https://github.com/keeganhk/PointMCD)
* **Group multi-view transformer for 3d shape analysis with spatial encoding**, IEEE Transactions on Multimedia 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10509815) [ :octocat: ](https://github.com/bigdata-graph/GMViT)

### Other Tasks
* **Multi-view teacher–student network**, Neural Networks 2022, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S0893608021004305)
* **Visual-Policy Learning Through Multi-Camera View to Single-Camera View Knowledge Distillation for Robot Manipulation Tasks**, IEEE Robotics and Automation Letters 2023, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10327777)
* **Structured knowledge distillation towards efficient multi-view 3d object detection**, BMVC 2023, [ :link: ](https://papers.bmvc2023.org/0339.pdf)
* **Kd-mvs: Knowledge distillation based self-supervised learning for multi-view stereo**, ECCV 2022, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-031-19821-2_36) [ :octocat: ](https://github.com/megvii-research/KD-MVS)
* **Multi-view knowledge distillation for efficient semantic segmentation**, Journal of Real-Time Image Processing 2023, [ :link: ](https://link.springer.com/article/10.1007/s11554-023-01296-6) 
* **Multi-view knowledge distillation transformer for human action recognition**, ArXiv 2023, [ :link: ](https://arxiv.org/abs/2303.14358)
* **Mt-mv-kdf: A novel multi-task multi-view knowledge distillation framework for myocardial infarction detection and localization**, Biomedical Signal Processing and Control 2024, [ :link: ](https://www.sciencedirect.com/science/article/pii/S1746809424004403)













## Text
### Neural Machine Translation (NMT)
* **Align-to-Distill: Trainable Attention Alignment for Knowledge Distillation in Neural Machine Translation**, arXiv 2024, [ :link: ](https://arxiv.org/abs/2403.01479) [ :octocat: ](https://github.com/ncsoft/Align-to-Distill)
* **Multi-Teacher Distillation With Single Model for Neural Machine Translation**, IEEE/ACM 2022, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9722996) [ :octocat: ](https://github.com/dropreg/DataPipe)
* **Selective Knowledge Distillation for Neural Machine Translation**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2105.12967) [ :octocat: ](https://github.com/LeslieOverfitting/selective_distillation)
* **Multilingual Neural Machine Translation with Knowledge Distillation**, ICLR 2019, [ :link: ](https://arxiv.org/abs/1902.10461) [ :octocat: ](https://github.com/RayeRen/multilingual-kd-pytorch)
* **Sequence-Level Knowledge Distillation**, Conference on empirical methods in natural language processing 2016, [ :link: ](https://aclanthology.org/D16-1139.pdf) [ :octocat: ](https://github.com/harvardnlp/seq2seq-attn)

### Question Answering (QA)
* **DPAL-BERT: A Faster and Lighter Question Answering Model**, CMES-Computer Modeling in Engineering & Sciences 2024, [ :link: ](https://www.researchgate.net/profile/Wenfeng-Zheng/publication/382518664_DPAL-BERT_A_Faster_and_Lighter_Question_Answering_Model/links/66a1d3705919b66c9f687e71/DPAL-BERT-A-Faster-and-Lighter-Question-Answering-Model.pdf)
* **Distilling Knowledge from Reader to Retriever for Question Answering**, ICLR 2021, [ :link: ](https://arxiv.org/abs/2012.04584) [ :octocat: ](https://github.com/facebookresearch/FiD)
* **Machine Reading Comprehension as Data Augmentation: A Case Study on Implicit Event Argument Extraction**, Conference on Empirical Methods in Natural Language Processing 2021, [ :link: ](https://aclanthology.org/2021.emnlp-main.214/) [ :octocat: ](https://github.com/jianliu-ml/DocMRC)
* **Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System**, ACM 2020, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3336191.3371792)
* **Attention-Guided Answer Distillation for Machine Reading Comprehension**, arXiv 2018, [ :link: ](https://arxiv.org/abs/1808.07644)

### Text Generation
* **Distilling Knowledge Learned in BERT for Text Generation**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1911.03829) [ :octocat: ](https://github.com/ChenRocks/Distill-BERT-Textgen)
* **TextKD-GAN: Text Generation Using Knowledge Distillation and Generative Adversarial Networks**, arXiv 2019, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-030-18305-9_9)
### Event Detection
* **Lifelong Event Detection with Knowledge Transfer**, EMNLP 2021, [ :link: ](https://aclanthology.org/2021.emnlp-main.428/) [ :octocat: ](https://github.com/perfec-yu/lifelong-ed)
* **Transferring Knowledge Distillation for Multilingual Social Event Detection**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2108.03084)
* **Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection**, AAAI 2019, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/4649) 
### Document Retrieval
* **Simplified TinyBERT: Knowledge Distillation for Document Retrieval**, ECIR 2021, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-030-72240-1_21) [ :octocat: ](https://github.com/cxa-unique/Simplified-TinyBERT)
* **Knowledge Distillation in Document Retrieval**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1911.11065)
### Text Recognition
* **Text Is Text, No Matter What: Unifying Text Recognition Using Knowledge Distillation**, ICCV 2021, [ :link: ](https://openaccess.thecvf.com/content/ICCV2021/html/Bhunia_Text_Is_Text_No_Matter_What_Unifying_Text_Recognition_Using_ICCV_2021_paper.html)
* **Joint architecture and knowledge distillation in CNN for Chinese text recognition**, Pattern Recognition 2021, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S0031320320305252)
### Named Entity Recognition (NER)
* **Discrepancy and Uncertainty Aware Denoising Knowledge Distillation for Zero-Shot Cross-Lingual Named Entity Recognition**, AAAI 2024, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/29762)
* **Reinforced Iterative Knowledge Distillation for Cross-Lingual Named Entity Recognition**, ACM 2021, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3447548.3467196)
### Text Summarization
* **Improving Neural Cross-Lingual Abstractive Summarization via Employing Optimal Transport Distance for Knowledge Distillation**, AAAI 2022, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/21359)
* **Noisy Self-Knowledge Distillation for Text Summarization**, arXiv 2020, [ :link: ](https://arxiv.org/abs/2009.07032) [ :octocat: ](https://github.com/nlpyang/NoisySumm)
* **An Information Distillation Framework for Extractive Summarization**, IEEE/ACM 2017, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8074745)
### Natural Language Understanding (NLU)
* **Alexa Teacher Model: Pretraining and Distilling Multi-Billion-Parameter Encoders for Natural Language Understanding Systems**, ACM 2022, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3534678.3539173) [ :octocat: ](https://github.com/amazon-science/alexa-teacher-models)
* **Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1903.12136)
* **Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1904.09482) [ :octocat: ](https://github.com/namisan/mt-dnn)
### Sentiment Analysis
* **Efficient sentiment analysis and topic modeling in nlp using knowledge distillation and transfer learning**, 2023, [ :link: ](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1795316)
* **BERT-Based Sentiment Analysis Using Distillation**, International conference on statistical language and speech processing 2020, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-030-59430-5_5)
### Text Classification
* **Data Distillation for Text Classification**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2104.08448) [ :octocat: ](https://github.com/liyongqi67/Data-Distillation-for-Text-Classification)
* **Cross-lingual Distillation for Text Classification**, arXiv 2017, [ :link: ](https://arxiv.org/abs/1705.02073) [ :octocat: ](https://github.com/xrc10/cross-distill)

## Speech
### Speech Recognition (ASR)
* **Cuing without sharing: A federated cued speech recognition framework via mutual knowledge distillation**, ACM 2023, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3581783.3612134) [ :octocat: ](https://github.com/YuxuanZHANG0713/FedCSR)
* **Knowledge Distillation-Based Training of Speech Enhancement for Noise-Robust Automatic Speech Recognition**, IEEE/ACM 2023, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10535505)
* **End-to-end emotional speech recognition using acoustic model adaptation based on knowledge distillation**, Springer Nature 2023, [ :link: ](https://link.springer.com/article/10.1007/s11042-023-14680-y)
* **DQ-Whisper: Joint Distillation and Quantization for Efficient Multilingual Speech Recognition**, arXiv 2023, [ :link: ](https://arxiv.org/abs/2305.10788)
 * **Knowledge distillation via module replacing for automatic speech recognition with recurrent neural network transducer**, Interspeech 2022, [ :link: ](https://par.nsf.gov/servlets/purl/10367863)
* **Inter-kd: Intermediate knowledge distillation for ctc-based automatic speech recognition**, IEEE SLT 2022, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10022581)
* **Mutual-learning sequence-level knowledge distillation for automatic speech recognition**, Neurocomputing 2021, [ :link: ](https://www.sciencedirect.com/science/article/pii/S0925231220318129)
* **Knowledge Distillation for Improved Accuracy in Spoken Question Answering**, ICASSP 2021, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9414999)
* **TutorNet: Towards flexible knowledge distillation for end-to-end speech recognition**, IEEE/ACM 2021 , [ :link: ](https://ieeexplore.ieee.org/abstract/document/9398543)
* **Cross-modal knowledge distillation method for automatic cued speech recognition**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2106.13686)
* **Knowledge Distillation from Offline to Streaming RNN Transducer for End-to-end Speech Recognition**, Interspeech 2020, [ :link: ](http://www.interspeech2020.org/uploadfile/pdf/Wed-1-5-3.pdf)
* **Distilling the Knowledge of BERT for Sequence-to-Sequence ASR**, arXiv 2020, [ :link: ](https://arxiv.org/abs/2008.03822) [ :octocat: ](https://github.com/hfutami/distill-bert-for-seq2seq-asr)
* **Knowledge Distillation for End-to-End Monaural Multi-talker ASR System**, Interspeech 2019, [ :link: ](https://www.isca-archive.org/interspeech_2019/zhang19i_interspeech.pdf)
* **Sequence-level knowledge distillation for model compression of attention-based sequence-to-sequence speech recognition**, ICASSP 2019, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8683171)
* **Essence Knowledge Distillation for Speech Recognition**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1906.10834)
* **Improved Knowledge Distillation from Bi-Directional to Uni-Directional LSTM CTC for End-to-End Speech Recognition**, IEEE SLT 2018, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8639629)
* **An Investigation of a Knowledge Distillation Method for CTC Acoustic Models**, ICASSP 2018, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8461995)
* **Distilling knowledge from ensembles of neural networks for speech recognition**, Interspeech 2016, [ :link: ](https://www.isca-archive.org/interspeech_2016/chebotar16_interspeech.pdf)
* **Robust Speech Recognition using Generalized Distillation Framework**, Interspeech 2016, [ :link: ](https://d1wqtxts1xzle7.cloudfront.net/79206022/Intersp_16-libre.pdf?1642728149=&response-content-disposition=inline%3B+filename%3DRobust_Speech_Recognition_Using_Generali.pdf&Expires=1740836183&Signature=XilAi7XeocBuU1ZX9xEJLw7rwqTPwS~c39cvCAswpULr8L8ueel5b3a9vUSgWrZIgWynq-OYyd6-rULE5UiFgy91LAvzdcjZnpp8mszVvvnsgr-lTPLxtA~Ytr4SXG8xlp7iUymo8cR6HRFn9SnwFGgY3evFrQsvgocFXeGk5MWjm1jf4F9qzeMd1S2XL1LVaw0e2Z7nFPMrgSQRy8wWZMvD70tHJfkzU-U-4nWQ1SZRLVMn9-k1LZd8z5IXqR800MZUTBQpkhUs0cAuHfofghVkV~MGG5aO4g-4SIFj9i5ROOWiWqgJUR40RlsXMIUZ-x5Nfaa0LacLTMB7H2q0LQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
### Speech Enhancement
* **Leveraging Non-Causal Knowledge via Cross-Network Knowledge Distillation for Real-Time Speech Enhancement**, IEEE Signal Processing Letters 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10502149)
* **MetricGAN-OKD: Multi-Metric Optimization of MetricGAN via Online Knowledge Distillation for Speech Enhancement**, ICML 2023, [ :link: ](https://proceedings.mlr.press/v202/shin23b.html) [ :octocat: ](https://github.com/wooseok-shin/MetricGAN-OKD)
* **Fast Real-time Personalized Speech Enhancement: End-to-End Enhancement Network (E3Net) and Knowledge Distillation**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2204.00771)
* **Test-Time Adaptation Toward Personalized Speech Enhancement: Zero-Shot Learning with Knowledge Distillation**, IEEE WASPAA 2021, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9632771) [ :octocat: ](https://github.com/kimsunwiub/PSE_ZeroShot_KD)
* **Speech Enhancement Using Generative Adversarial Network by Distilling Knowledge from Statistical Method**, Applied Sciences 2019, [ :link: ](https://www.mdpi.com/2076-3417/9/16/3396)
* **Student-teacher network learning with enhanced features**, ICASSP 2017, [ :link: ](https://ieeexplore.ieee.org/abstract/document/7953163)
* **Sub-Band Knowledge Distillation Framework for Speech Enhancement**, arXiv 2005, [ :link: ](https://arxiv.org/abs/2005.14435)
### Speaker Recognition and Verification
* **Integrating Voice Activity Detection to Enhance Robustness of On-Device Speaker Verification**, PRICAI 2024, [ :link: ](https://link.springer.com/chapter/10.1007/978-981-96-0125-7_31)
* **Emphasized Non-Target Speaker Knowledge in Knowledge Distillation for Automatic Speaker Verification**, ICASSP 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10447160) [ :octocat: ](https://github.com/ductuantruong/enskd)
* **Cross-Modal Distillation for Speaker Recognition**, AAAI 2023, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/26525)
* **Class token and knowledge distillation for multi-head self-attention speaker verification systems**, Digital Signal Processing 2023, [ :link: ](https://www.sciencedirect.com/science/article/pii/S1051200422004766)
* **Label-free Knowledge Distillation with Contrastive Loss for Light-weight Speaker Recognition**, ISCSLP 2022, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10038276)
* **Self-Knowledge Distillation via Feature Enhancement for Speaker Verification**, ICASSP 2022, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9746529)
* **Knowledge Distillation and Random Erasing Data Augmentation for Text-Dependent Speaker Verification**, ICASSP 2020, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9053153)
### Speech Translation
* **CKDST: Comprehensively and Effectively Distill Knowledge from Machine Translation to End-to-End Speech Translation**, ACL 2023, [ :link: ](https://aclanthology.org/2023.findings-acl.195/) [ :octocat: ](https://github.com/ethanyklei/CKDST)
* **Source and Target Bidirectional Knowledge Distillation for End-to-end Speech Translation**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2104.06457)
* **End-to-End Speech-Translation with Knowledge Distillation: FBK@IWSLT2020**, arXiv 2020, [ :link: ](https://arxiv.org/abs/2006.02965)
* **End-to-End Speech Translation with Knowledge Distillation**, arXiv 2019, [ :link: ](https://arxiv.org/abs/1904.08075)
### Speech Synthesis (Text-to-Speech)
* **DMOSpeech: Direct Metric Optimization via Distilled Diffusion Model in Zero-Shot Speech Synthesis**, arXiv 2024, [ :link: ](https://arxiv.org/abs/2410.11097)
* **LRSpeech: Extremely Low-Resource Speech Synthesis and Recognition**, ACM 2020, [ :link: ](https://dl.acm.org/doi/abs/10.1145/3394486.3403331)
* **Parallel WaveNet: Fast High-Fidelity Speech Synthesis**, ICML 2018, [ :link: ](https://proceedings.mlr.press/v80/oord18a.html)
### Speech Separation
* **Teacher-Student MixIT for Unsupervised and Semi-supervised Speech Separation**, arXiv 2021, [ :link: ](https://arxiv.org/abs/2106.07843)
* **Distilled Binary Neural Network for Monaural Speech Separation**, IJCNN 2018, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8489456)
### Spoken Language Identification and Understanding
* **DiffSLU: Knowledge Distillation Based Diffusion Model for Cross-Lingual Spoken Language Understanding**, Interspeech 2023, [ :link: ](https://www.isca-archive.org/interspeech_2023/mao23_interspeech.pdf)
* **Sequence-Level Knowledge Distillation for Class-Incremental End-to-End Spoken Language Understanding**, arXiv 2023, [ :link: ](https://arxiv.org/abs/2305.13899)
* **An Investigation of the Combination of Rehearsal and Knowledge Distillation in Continual Learning for Spoken Language Understanding**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2211.08161) [ :octocat: ](https://github.com/umbertocappellazzo/CL_SLU)
* **Two-Stage Textual Knowledge Distillation for End-to-End Spoken Language Understanding**, ICASSP 2021, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9414619) [ :octocat: ](https://github.com/clovaai/textual-kd-slu)
* **Knowledge Distillation-Based Representation Learning for Short-Utterance Spoken Language Identification**, IEEE/ACM 2020, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9195801)
* **Interactive Learning of Teacher-student Model for Short Utterance Spoken Language Identification**, ICASSP 2019, [ :link: ](https://ieeexplore.ieee.org/abstract/document/8683371)
* **Feature Representation of Short Utterances based on Knowledge Distillation for Spoken Language Identification**, Interspeech 2018, [ :link: ](https://www.researchgate.net/profile/Sheng-Li-60/publication/327389039_Feature_Representation_of_Short_Utterances_Based_on_Knowledge_Distillation_for_Spoken_Language_Identification/links/5bbad5164585159e8d8be2a7/Feature-Representation-of-Short-Utterances-Based-on-Knowledge-Distillation-for-Spoken-Language-Identification.pdf)
### Deepfake Speech and Spoofing Detection
* **Audio Deepfake Detection: A Continual Approach with Feature Distillation and Dynamic Class Rebalancing**, International Conference on Pattern Recognition 2025, [ :link: ](https://link.springer.com/chapter/10.1007/978-3-031-78305-0_14)
* **One-Class Knowledge Distillation for Spoofing Speech Detection**, ICASSP 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10446270)
* **Learning From Yourself: A Self-Distillation Method For Fake Speech Detection**, ICASSP 2023, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10096837)
* **Lightweight Voice Spoofing Detection Using Improved One-Class Learning and Knowledge Distillation**, IEEE Transactions on Multimedia 2023, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10269071)
* **A voice spoofing detection framework for IoT systems with feature pyramid and online knowledge distillation**, Journal of Systems Architecture 2023, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S1383762123001601)
* **Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**, arXiv 2022, [ :link: ](https://arxiv.org/abs/2203.17031)
### Audio Classification and Tagging
* **Enhanced Feature Learning with Normalized Knowledge Distillation for Audio Tagging**, Interspeech 2024, [ :link: ](https://www.isca-archive.org/interspeech_2024/tang24b_interspeech.pdf)
* **CED: Consistent Ensemble Distillation for Audio Tagging**, ICASSP 2024, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10446348) [ :octocat: ](https://github.com/RicherMans/ced)
* **Efficient Large-Scale Audio Tagging Via Transformer-to-CNN Knowledge Distillation**, ICASSP 2023, [ :link: ](https://ieeexplore.ieee.org/abstract/document/10096110) [ :octocat: ](https://github.com/fschmid56/EfficientAT)
* **Joint framework with deep feature distillation and adaptive focal loss for weakly supervised audio tagging and acoustic event detection**, Digital Signal Processing 2022, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S105120042200063X)
* **Enhanced Audio Tagging via Multi- to Single-Modal Teacher-Student Mutual Learning**, AAAI 2021, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/17280)
* **Intra-Utterance Similarity Preserving Knowledge Distillation for Audio Tagging**, arXiv 2020, [ :link: ](https://arxiv.org/abs/2009.01759)
### Spoken Question Answering and Conversational AI
* **Towards Data Distillation for End-to-end Spoken Conversational Question Answering**, ICLR 2021, [ :link: ](https://arxiv.org/abs/2010.08923)
* **Knowledge Distillation for Improved Accuracy in Spoken Question Answering**, ICASSP 2021, [ :link: ](https://ieeexplore.ieee.org/abstract/document/9414999)
* **MRD-Net: Multi-Modal Residual Knowledge Distillation for Spoken Question Answering**, IJCAI 2021, [ :link: ](https://www.ijcai.org/proceedings/2021/0549.pdf)
* **Contextualized Attention-based Knowledge Transfer for Spoken Conversational Question Answering**, arXiv 2020, [ :link: ](https://arxiv.org/abs/2010.11066)
### Audio Captioning and Retrieval
* **Efficient Audio Captioning with Encoder-Level Knowledge Distillation**, arXiv 2024, [ :link: ](https://arxiv.org/abs/2407.14329)
* **A Knowledge Distillation Approach to Improving Language-Based Audio Retrieval Models**, DCASE2024 Challenge, [ :link: ](https://dcase.community/documents/challenge2024/technical_reports/DCASE2024_Primus_76_t8.pdf) [ :octocat: ](https://github.com/OptimusPrimus/salsa)




## Video
* **Knowledge Distillation in Video-Based Human Action Recognition: An Intuitive Approach to Efficient and Flexible Model Training**, Journal of Imaging 2024, [ :link: ](https://www.mdpi.com/2313-433X/10/4/85)
* **Multi-teacher knowledge distillation for compressed video action recognition based on deep learning**, Journal of systems architecture 2020, [ :link: ](https://www.sciencedirect.com/science/article/abs/pii/S1383762119305028)
* **Generative Model-Based Feature Knowledge Distillation for Action Recognition**, AAAI 2024, [ :link: ](https://ojs.aaai.org/index.php/AAAI/article/view/29473) [ :octocat: ](https://github.com/aaai-24/Generative-based-KD)
* **Advancing Compressed Video Action Recognition through Progressive Knowledge Distillation**, arxive 2024, [ :link: ](https://arxiv.org/abs/2407.02713) [ :octocat: ](https://github.com/Efstathia-Soufleri/PKD)
* **Attention Distillation for Learning Video Representations**, arxive 2019, [ :link: ](https://aptx4869lm.github.io/AttentionDistillation/)
* **The Staged Knowledge Distillation in Video Classification: Harmonizing Student Progress by a Complementary Weakly Supervised Framework**, IEEE Transactions on Circuits and Systems for Video Technology 2024, [ :link: ](https://ieeexplore.ieee.org/document/10182291)
* **Efficient Video Classification Using Fewer Frames**, CVPR 2019, [ :link: ](https://openaccess.thecvf.com/content_CVPR_2019/html/Bhardwaj_Efficient_Video_Classification_Using_Fewer_Frames_CVPR_2019_paper.html)
* **MobileVOS: Real-Time Video Object Segmentation Contrastive Learning meets Knowledge Distillation**, CVPR 2024, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Miles_MobileVOS_Real-Time_Video_Object_Segmentation_Contrastive_Learning_Meets_Knowledge_Distillation_CVPR_2023_paper.html)
* **Let Video Teaches You More: Video-to-Image Knowledge Distillation using DEtection TRansformer for Medical Video Lesion Detection**, IEEE International Conference on Bioinformatics and Biomedicine (BIBM) 2024, [ :link: ](https://www.computer.org/csdl/proceedings-article/bibm/2024/10822332/23oo174OOd2)
* **Offline-to-Online Knowledge Distillation for Video Instance Segmentation**, WACV 2024, [ :link: ](https://openaccess.thecvf.com/content/WACV2024/html/Kim_Offline-to-Online_Knowledge_Distillation_for_Video_Instance_Segmentation_WACV_2024_paper.html) 
* **Dual Learning with Dynamic Knowledge Distillation for Partially Relevant Video Retrieval**, ICCV 2023, [ :link: ](https://openaccess.thecvf.com/content/ICCV2023/html/Dong_Dual_Learning_with_Dynamic_Knowledge_Distillation_for_Partially_Relevant_Video_ICCV_2023_paper.html) [ :octocat: ](https://github.com/HuiGuanLab/DL-DKD)
*  **How many Observations are Enough? Knowledge Distillation for Trajectory Forecasting**, CVPR 2022, [ :link: ](https://openaccess.thecvf.com/content/CVPR2022/html/Monti_How_Many_Observations_Are_Enough_Knowledge_Distillation_for_Trajectory_Forecasting_CVPR_2022_paper.html)
*  **Mask Again: Masked Knowledge Distillation for Masked Video Modeling**, ACM 2023, [ :link: ](https://dl.acm.org/doi/10.1145/3581783.3612129) [ :octocat: ](https://github.com/xiaojieli0903/MaskAgain)
*  **Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning**, CVPR 2023, [ :link: ](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Masked_Video_Distillation_Rethinking_Masked_Feature_Modeling_for_Self-Supervised_Video_CVPR_2023_paper.html) [ :octocat: ](https://github.com/ruiwang2021/mvd)
*  **RankDVQA-mini: Knowledge distillation-driven deep video quality assessment**, Picture Coding Symposium (PCS) 2024, [ :link: ](https://ieeexplore.ieee.org/document/10566364) [ :octocat: ](https://chenfeng-bristol.github.io/RankDVQA-mini/)
*  **Ultrafast Video Attention Prediction with Coupled Knowledge Distillation**, AAAI 2020, [ :link: ](https://cdn.aaai.org/ojs/6710/6710-13-9939-1-10-20200522.pdf)
*  **Online Model Distillation for Efficient Video Inference**, ICCV 2019, [ :link: ](https://openaccess.thecvf.com/content_ICCV_2019/html/Mullapudi_Online_Model_Distillation_for_Efficient_Video_Inference_ICCV_2019_paper.html)
*  **VideoAdviser: Video Knowledge Distillation for Multimodal Transfer Learning**, IEEE Access 2023, [ :link: ](https://ieeexplore.ieee.org/document/10136716?denied=)
*  **Asr is all you need: Cross-modal distillation for lip reading**, ICASSP 2020, [ :link: ](https://arxiv.org/pdf/1911.12747)
*  **Audio–Visual Model Distillation Using Acoustic Images**, WACV 2020, [ :link: ](https://openaccess.thecvf.com/content_WACV_2020/html/Perez_Audio-Visual_Model_Distillation_Using_Acoustic_Images_WACV_2020_paper.html)
*  **Spatio-Temporal Graph for Video Captioning with Knowledge Distillation**, CVPR 2020, [ :link: ](https://openaccess.thecvf.com/content_CVPR_2020/html/Pan_Spatio-Temporal_Graph_for_Video_Captioning_With_Knowledge_Distillation_CVPR_2020_paper.html)
