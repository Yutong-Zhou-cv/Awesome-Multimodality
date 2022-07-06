# <p align=center> Awesome Multimodality 🎶📜</p>

<div align=center>

<p>
 
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) 
 ![GitHub stars](https://img.shields.io/github/stars/Yutong-Zhou-cv/Awesome-Multimodality.svg?color=red&style=for-the-badge) 
 ![GitHub forks](https://img.shields.io/github/forks/Yutong-Zhou-cv/Awesome-Multimodality.svg?color=yellow&style=for-the-badge) 
 ![GitHub activity](https://img.shields.io/github/last-commit/Yutong-Zhou-cv/Awesome-Multimodality?style=for-the-badge) 
 ![Visitors](https://visitor-badge.glitch.me/badge?page_id=Yutong-Zhou-cv/Awesome-Multimodality) 

</p>

A collection of resources on multimodal learning research.
 
</div>

## <span id="head-content"> *Content* </span>
* - [ ] [1. Description](#head1)
* - [ ] [2. Topic Order](#head2)
  * - [ ] [Survey](#head-Survey)
  * - [ ] [👑 Dataset](#head-dataset)
  * - [ ] [💬 Vision and language Pre-training (VLP)](#head-VLP)
* - [ ] [3. Chronological Order](#head3)
  * - [ ] [Survey](#head-Survey)
  * - [ ] [2022](#head-2022)
  * - [ ] [2021](#head-2021)
  * - [ ] [2020](#head-2020)

* - [ ] [4. Courses](#head4)

* [*Contact Me*](#head5)

## <span id="head1"> *1.Description* </span>

>🐌 Markdown Format:
>
> * (Conference/Journal Year) **Title**, First Author et al. [[Paper](URL)] [[Code](URL)] [[Project](URL)] <br/>
> * (Conference/Journal Year) [💬Topic] **Title**, First Author et al. [[Paper](URL)] [[Code](URL)] [[Project](URL)]
>     * (Optional) ```🌱``` or ```📌 ```
>     * (Optional) 🚀 or 👑 or 📚

* ```🌱: Novel idea```
* ```📌: The first...```
* 🚀: State-of-the-Art
* 👑: Novel dataset/model
* 📚：Downstream Tasks 

## <span id="head2"> *2. Topic Order* </span>

* <span id="head-Survey"> **[Survey](https://github.com/Yutong-Zhou-cv/Awesome-Survey-Papers)**  </span>
    * (arXiv preprint 2022) [💬Vision and language Pre-training (VLP)] **Vision-and-Language Pretraining**, Thong Nguyen et al. [[v1](https://arxiv.org/abs/2207.01772)](2022.07.05)
    * (arXiv preprint 2022) [💬Video Saliency Detection] **A Comprehensive Survey on Video Saliency Detection with Auditory Information: the Audio-visual Consistency Perceptual is the Key!**, Chenglizhao Chen et al. [[v1](https://arxiv.org/abs/2206.13390)](2022.06.20)
    * (arXiv preprint 2022) [💬Transformer] **Multimodal Learning with Transformers: A Survey**, Peng Xu et al. [[v1](https://arxiv.org/abs/2206.06488)](2022.06.13)
    * (arXiv preprint 2022) [💬Vision and language Pre-training (VLP)] **VLP: A Survey on Vision-Language Pre-training**, Feilong Chen et al. [[v1](https://arxiv.org/abs/2202.09061v1)](2022.02.18) [[v2](https://arxiv.org/abs/2202.09061v2)](2022.02.21) 
    * (arXiv preprint 2022) [💬Vision and language Pre-training (VLP)] **A Survey of Vision-Language Pre-Trained Models**, Yifan Du et al. [[v1](https://arxiv.org/abs/2202.10936)](2022.02.18) 
    * (arXiv preprint 2022) [💬Multi-Modal Knowledge Graph] **Multi-Modal Knowledge Graph Construction and Application: A Survey**, Xiangru Zhu et al. [[v1](https://arxiv.org/pdf/2202.05786.pdf)](2022.02.11) 
    * (arXiv preprint 2021) **A Survey on Multi-modal Summarization**, Anubhav Jangra et al. [[v1](https://arxiv.org/pdf/2109.05199.pdf)](2021.09.11) 
    * (Information Fusion 2021) [💬Vision and language] **Multimodal research in vision and language: A review of current and emerging trends**, ShagunUppal et al. [[v1](https://www.sciencedirect.com/science/article/pii/S1566253521001512)](2021.08.01) 

* <span id="head-dataset"> **👑 Dataset**  </span>
    * (arXiv preprint 2022) **Wukong: 100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework**, Jiaxi Gu et al. [[Paper](https://arxiv.org/abs/2202.06767)] [[Download](https://wukong-dataset.github.io/wukong-dataset/download.html)]
        * The Noah-Wukong dataset is a large-scale multi-modality Chinese dataset.
        * The dataset contains 100 Million <image, text> pairs
        * Images in the datasets are filtered according to the size ( > 200px for both dimensions ) and aspect ratio ( 1/3 ~ 3 )
        * Text in the datasets are filtered according to its language, length and frequency. Privacy and sensitive words are also taken into consideration.
    * (arXiv preprint 2022) **WuDaoMM: A large-scale Multi-Modal Dataset for Pre-training models**, Sha Yuan et al. [[Paper](https://arxiv.org/abs/2203.11480)] [[Download](https://github.com/BAAI-WuDao/WuDaoMM/)]

* <span id="head-VLP"> **💬 Vision and language Pre-training (VLP)**  </span>
    * ⭐⭐[**CVPR 2022 Tutorial**] **Recent Advances in Vision-and-Language Pre-training** [[Project](https://vlp-tutorial.github.io/2022/)] 
    * ⭐⭐(arXiv preprint 2022) [💬Data Augmentation] **MixGen: A New Multi-Modal Data Augmentation**, Xiaoshuai Hao et al. [[Paper](https://arxiv.org/abs/2206.08358)]
      * 📚 Downstream Tasks: Image-Text Retrieval, Visual Question Answering (VQA), Visual Grounding, Visual Reasoning, Visual Entailment
    * ⭐⭐(ICML 2022) **Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts**, Yan Zeng et al. [[Paper](https://arxiv.org/abs/2111.08276)] [[Code](https://github.com/zengyan-97/X-VLM)]
      * 🚀 SOTA(2022/06/16): Cross-Modal Retrieval on COCO 2014 & Flickr30k, Visual Grounding on RefCOCO+ val & RefCOCO+ testA, RefCOCO+ testB 
      * 📚 Downstream Tasks: Image-Text Retrieval, Visual Question Answering (VQA), Natural Language for Visual Reasoning (NLVR2), Visual Grounding, Image Captioning
    * ⭐⭐(arXiv preprint 2022) **Multimodal Contrastive Learning with LIMoE: the Language-Image Mixture of Experts**, Basil Mustafa et al. [[Paper](https://arxiv.org/abs/2206.02770)] [[Blog](https://ai.googleblog.com/2022/06/limoe-learning-multiple-modalities-with.html)]
      * 📌 LIMoE: The first large-scale multimodal mixture of experts models.
    * (CVPR 2022) **Unsupervised Vision-and-Language Pre-training via Retrieval-based Multi-Granular Alignment**, Mingyang Zhou et al. [[Paper](https://arxiv.org/abs/2203.00242)] [[Code](https://github.com/zmykevin/UVLP)]
      * 📚 Downstream Tasks: Visual Question Answering(VQA), Natural Language for Visual reasoning(NLVR2), Visual Entailment, Referring Expression(RefCOCO+)    
    * ⭐(arXiv preprint 2022) **One Model, Multiple Modalities: A Sparsely Activated Approach for Text, Sound, Image, Video and Code**, Yong Dai et al. [[Paper](https://arxiv.org/abs/2205.06126)]
      * 📚 Downstream Tasks: Text Classification, Automatic Speech Recognition, Text-to-Image Retrieval, Text-to-Video Retrieval, Text-to-Code Retrieval
    * (arXiv preprint 2022) **Zero and R2D2: A Large-scale Chinese Cross-modal Benchmark and A Vision-Language Framework**, Chunyu Xie et al. [[Paper](https://arxiv.org/abs/2205.03860)]
      * 📚 Downstream Tasks: Image-text Retrieval, Chinese Image-text matching
    * (arXiv preprint 2022) **Vision-Language Pre-Training with Triple Contrastive Learning**, Jinyu Yang et al. [[Paper](https://arxiv.org/abs/2202.10401)] [[Code](https://github.com/uta-smile/TCL)]
      * 📚 Downstream Tasks: Image-text Retrieval, Visual Question Answering, Visual Entailment, Visual Reasoning
    * (arXiv preprint 2022) **MVP: Multi-Stage Vision-Language Pre-Training via Multi-Level Semantic Alignment**, Zejun Li et al. [[Paper](https://arxiv.org/abs/2201.12596)]
      * 📚 Downstream Tasks: Image-text Retrieval, Multi-Modal Classification, Visual Grounding
    * (arXiv preprint 2022) **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation**, Junnan Li et al. [[Paper](https://arxiv.org/abs/2201.12086)] [[Code](https://github.com/salesforce/BLIP)]
      * 📚 Downstream Tasks: Image-text Retrieval, Image Captioning, Visual Question Answering, Visual Reasoning, Visual Dialog
    * (ICML 2021) **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision**, Wonjae Kim et al. [[Paper](https://arxiv.org/abs/2102.03334)]
      * 📚 Downstream Tasks: Image Text Matching, Masked Language Modeling


## <span id="head3"> *3. Chronological Order* </span>

* <span id="head-2022"> **2022**  </span>
    * (AI Ethics and Society 2022) [💬Multi-modal & Bias] **American == White in Multimodal Language-and-Image AI**, Robert Wolfe et al. [[Paper](https://arxiv.org/abs/2207.00691)] 
    * (Interspeech 2022) [💬Audio-Visual Speech Separation] **Multi-Modal Multi-Correlation Learning for Audio-Visual Speech Separation**, Xiaoyu Wang et al. [[Paper](https://arxiv.org/abs/2207.01197)] 
    * (arXiv preprint 2022) [💬Multi-modal for Recommendation] **Personalized Showcases: Generating Multi-Modal Explanations for Recommendations**, An Yan et al. [[Paper](https://arxiv.org/abs/2207.00422)] 
    * (CVPR 2022) [💬Video Synthesis] **Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning**, Ligong Han et al. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Han_Show_Me_What_and_Tell_Me_How_Video_Synthesis_via_CVPR_2022_paper.pdf)] [[Code](https://github.com/snap-research/MMVID)] [[Project](https://snap-research.github.io/MMVID/)]
    * (NAACL 2022) [💬Dialogue State Tracking] **Multimodal Dialogue State Tracking**, Hung Le et al. [[Paper](https://arxiv.org/abs/2206.07898)] 
    * (arXiv preprint 2022) [💬Multi-modal Multi-task] **MultiMAE: Multi-modal Multi-task Masked Autoencoders**, Roman Bachmann et al. [[Paper](https://arxiv.org/abs/2204.01678)] [[Code](https://github.com/EPFL-VILAB/MultiMAE)] [[Project](https://multimae.epfl.ch/)] 
    * (CVPR 2022) [💬Text-Video Retrieval] **X-Pool: Cross-Modal Language-Video Attention for Text-Video Retrieval**, Satya Krishna Gorti et al. [[Paper](https://arxiv.org/abs/2203.15086)] [[Code](https://github.com/layer6ai-labs/xpool)] [[Project](https://layer6ai-labs.github.io/xpool/)] 
    * (NAACL 2022 2022) [💬Visual Commonsense] **Visual Commonsense in Pretrained Unimodal and Multimodal Models**, Chenyu Zhang et al. [[Paper](https://arxiv.org/abs/2205.01850)] [[Code](https://github.com/ChenyuHeidiZhang/VL-commonsense)]
    * (arXiv preprint 2022) [💬Pretraining framework] **i-Code: An Integrative and Composable Multimodal Learning Framework**, Ziyi Yang et al. [[Paper](https://arxiv.org/abs/2205.01818)]
    * (CVPR 2022) [💬Food Retrieval] **Transformer Decoders with MultiModal Regularization for Cross-Modal Food Retrieval**, Mustafa Shukor et al. [[Paper](https://arxiv.org/abs/2204.09730)] [[Code](https://github.com/mshukor/TFood)] 
    * (arXiv preprint 2022) [💬Image+Videos+3D Data Recognition] **Omnivore: A Single Model for Many Visual Modalities**, Rohit Girdhar et al. [[Paper](https://arxiv.org/abs/2201.08377)] [[Code](https://github.com/facebookresearch/omnivore)] [[Project](https://facebookresearch.github.io/omnivore/)]
    * (arXiv preprint 2022) [💬Hyper-text Language-image Model] **CM3: A Causal Masked Multimodal Model of the Internet**, Armen Aghajanyan et al. [[Paper](https://arxiv.org/abs/2201.07520)] 

* <span id="head-2021"> **2021**  </span>
    * (arXiv preprint 2021) [💬Visual Synthesis] **NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion**, Chenfei Wu et al. [[Paper](https://arxiv.org/abs/2111.12417)] [[Code](https://github.com/microsoft/NUWA)]
     ![Figure from paper](pic/NUWA.gif)
     > *(From: https://github.com/microsoft/NUWA [2021/11/30])*
    * (ICCV 2021) [💬Video-Text Alignment] **TACo: Token-aware Cascade Contrastive Learning for Video-Text Alignment**, Jianwei Yang et al. [[Paper](https://arxiv.org/abs/2108.09980)]
    * (arXiv preprint 2021) [💬Class-agnostic Object Detection] **Multi-modal Transformers Excel at Class-agnostic Object Detection**, Muhammad Maaz et al. [[Paper](https://arxiv.org/abs/2111.11430v1)] [[Code](https://github.com/mmaaz60/mvits_for_class_agnostic_od)]
    * (ACMMM 2021) [💬Video-Text Retrieval] **HANet: Hierarchical Alignment Networks for Video-Text Retrieval**, Peng Wu et al. [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475515)] [[Code](https://github.com/Roc-Ng/HANet)]
    * (ICCV 2021) [💬Video Recognition] **AdaMML: Adaptive Multi-Modal Learning for Efficient Video Recognition**, Rameswar Panda et al. [[Paper](https://rpand002.github.io/data/ICCV_2021_adamml.pdf)] [[Project](https://rpand002.github.io/adamml.html)] [[Code](https://github.com/IBM/AdaMML)]
    * (ICCV 2021) [💬Video Representation] **CrossCLR: Cross-modal Contrastive Learning For Multi-modal Video Representations**, Mohammadreza Zolfaghari et al. [[Paper](https://arxiv.org/abs/2109.14910)]
    * (ICCV 2021 **Oral**) [💬Text-guided Image Manipulation] **StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery**, Or Patashnik et al. [[Paper](https://arxiv.org/abs/2103.17249)] [[Code](https://github.com/orpatashnik/StyleCLIP)] [[Play](https://replicate.ai/orpatashnik/styleclip)]
    * (ICCV 2021) [💬Facial Editing] **Talk-to-Edit: Fine-Grained Facial Editing via Dialog**, Yuming Jiang et al. [[Paper](https://arxiv.org/abs/2109.04425)] [[Code](https://github.com/yumingj/Talk-to-Edit)] [[Project](https://www.mmlab-ntu.com/project/talkedit/)] [[Dataset Project](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebA_Dialog.html)] [[Dataset(CelebA-Dialog Dataset)](https://drive.google.com/drive/folders/18nejI_hrwNzWyoF6SW8bL27EYnM4STAs)] 
    * (arXiv preprint 2021) [💬Video Action Recognition] **ActionCLIP: A New Paradigm for Video Action Recognition**, Mengmeng Wang et al. [[Paper](https://arxiv.org/abs/2109.08472)] 

* <span id="head-2020"> **2020**  </span>
    * (EMNLP 2020) [💬Video+Language Pre-training] **HERO: Hierarchical Encoder for Video+Language Omni-representation Pre-training**, Linjie Li et al. [[Paper](https://arxiv.org/abs/2005.00200)] [[Code](https://github.com/linjieli222/HERO)]

## <span id="head4"> *3.Courses* </span>

* [CMU Multimodal Learning](https://cmu-multicomp-lab.github.io/mmml-course/fall2020/)

## <span id="head5"> *Contact Me* </span>

* [Yutong ZHOU](https://github.com/Yutong-Zhou-cv) in [Interaction Laboratory, Ritsumeikan University.](https://github.com/Rits-Interaction-Laboratory) ଘ(੭*ˊᵕˋ)੭

* If you have any question, please feel free to contact Yutong ZHOU (E-mail: <zhou@i.ci.ritsumei.ac.jp>).
