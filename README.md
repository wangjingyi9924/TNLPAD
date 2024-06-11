# TNLPAD

TPAMI: Tackling noisy labels with network parameter additive decomposition (PyTorch implementation).

This is the code for the paper: [Tackling noisy labels with network parameter additive decomposition](https://scholar.google.com/citations?view_op=view_citation&hl=zh-CN&user=jRsugY0AAAAJ&sortby=pubdate&citation_for_view=jRsugY0AAAAJ:TQgYirikUcIC)

Jingyi Wang, Xiaobo Xia, Long Lan, Xinghao Wu, Jun Yu, Wenjing Yang, Bo Han, Tongliang Liu.



## Dependencies

Ubuntu 18.04

Python 3.6

PyTorch, verion=1.4.0

CUDA, version=10.1



## Experiments
We verify the effectiveness of the proposed method on synthetic noisy datasets. In this repository, we provide the used [datasets](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) (the images and labels have been processed to .npy format). You should put the datasets in the folder “data” when you have downloaded them.
Training example:

```
python main.py --dataset mnist --noise_type symmetric --noise_rate 0.4 --c2 2 --seed 1
```

If you find this code useful in your research, please cite:

```
@ARTICLE{wang2024tackling,
  author={Wang, Jingyi and Xia, Xiaobo and Lan, Long and Wu, Xinghao and Yu, Jun and Yang, Wenjing and Han, Bo and Liu, Tongliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Tackling Noisy Labels With Network Parameter Additive Decomposition}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Noise measurement;Training;Additives;Robustness;Training data;Noise robustness;Upper bound;Early stopping;learning with noisy labels;memorization effect;parameter decomposition},
  doi={10.1109/TPAMI.2024.3382138}
}
```
