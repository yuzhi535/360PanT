# 360PanT (WACV-2025)

**360PanT: Training-Free Text-Driven 360-Degree Panorama-to-Panorama Translation**

[Hai Wang](https://littlewhitesea.github.io/) • [Jing-Hao Xue](https://www.homepages.ucl.ac.uk/~ucakjxu/)

[![Project](https://img.shields.io/badge/Project-Website-orange)](https://littlewhitesea.github.io/360PanT.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2409.08397-b31b1b.svg)](https://arxiv.org/abs/2409.08397)

### Jupyter Notebook and Datasets

| Link | Description
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/littlewhitesea/360PanT/blob/main/360PanT.ipynb) | Google Colab of 360PanT [which can be run on Nvidia T4 (16G)]
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/littlewhitesea/360PanT/blob/main/360PanT_(F).ipynb) | Google Colab of 360PanT (F) [which can be run on Nvidia L4 (24GB)]
[Datasets](https://drive.google.com/file/d/1L6-zczpGk08J8ex1-p3Leb8XfmjMg8Qu/view?usp=sharing) | 360PanoI-Pan2Pan and 360syn-Pan2Pan
[basis.zip](https://drive.google.com/file/d/1_ezbzljckjqg4Qjx9xfmaK4___xHTZrC/view?usp=sharing) | semantic bases of indoors and outdoors for FreeControl and 360PanT (F)

## Useful Tools

[360 panoramic images viewer](https://renderstuff.com/tools/360-panorama-web-viewer/): It could be used to view the synthesized 360-degree panorama.

[CLIP-score](https://github.com/OpenAI/CLIP): It contains Google Colab to calculate the CLIP-score.

[DINO-score](https://github.com/omerbt/Splice): It contains Google Colab to calculate the DINO-score.


## Citation
If you find the code helpful in your research or work, please cite our paper:
```Bibtex
@InProceedings{Wang_2025_WACV,
    author    = {Wang, Hai and Xue, Jing-Hao},
    title     = {360PanT: Training-Free Text-Driven 360-Degree Panorama-to-Panorama Translation},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {212-221}
}
```

## Acknowledgments

We thank [MultiDiffusion](https://github.com/omerbt/MultiDiffusion), [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play), [FreeControl](https://github.com/genforce/freecontrol) and [StitchDiffusion](https://github.com/littlewhitesea/StitchDiffusion). Our work is based on their excellent codes. 
