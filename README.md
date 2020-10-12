## Code for "A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives"

By [Alexander Mathis](https://github.com/AlexEMG) | [Steffen Schneider](https://github.com/stes) | [Jessy Lauer](https://github.com/jeylau) | [Mackenzie Mathis](https://github.com/MMathisLab)

Here we provide code that is used in the worked-examples we provide in our Primer on Deep Learning for Motion Capture.

Publication: TBA
Preprint: https://arxiv.org/abs/2009.00564

### Illustrating augmention methods (Figure 3)

Create the figure in Google Colaboratory: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepLabCut/Primer-MotionCapture/blob/master/COLAB_Primer_MotionCapture_Fig3.ipynb)

Or in plain Python:

```
python Illustrating-Augmentations-FigurePipelineMontBlancBirds.py
python Illustrating-Augmentations-FigurePipelineMouse.py
```

This code creates images like this:

<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/1593721822684-DJJIOEEC4YHRO2ZRW6Z3/ke17ZwdGBToddI8pDm48kMqyqP2xgoLMxs8NG4McAT9Zw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZamWLI2zvYWH8K3-s_4yszcp2ryTI0HqTOaaUohrI8PIT5G0e3JcWjRHcSN0vw_Zk3UAC_JV7w-s1xTX3wfPLu0/augs.png?format=1000w" width="55%">
</p>

### Labeling Pitfalls: How corruptions affect performance (Figure 7)

```
python CorruptionFigure.py
```
Creates figures like this:

<p align="center">
<img src="ResultsCorruption/LabelingCorruptionimpact_10Percent.png?format=1000w" width="55%">
</p>



More code to come, stay tuned!



Do you ant to contribute labeled data to the DeepLabCut project? Check out http://contrib.deeplabcut.org
