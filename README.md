## Code for "A Primer on Motion Capture with Deep Learning: Principles, Pitfalls and Perspectives"

By [Alexander Mathis](https://github.com/AlexEMG) | [Steffen Schneider](https://github.com/stes) | [Jessy Lauer](https://github.com/jeylau) | [Mackenzie Mathis](https://github.com/MMathisLab)

Here we provide code that we used in the worked-examples we provide in our Primer on Deep Learning for Motion Capture.

Publication in Neuron: https://www.cell.com/neuron/fulltext/S0896-6273(20)30717-0

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

To simulate inattentive labeling we corrupted the original dataset from [Mathis et al.](https://www.nature.com/articles/s41593-018-0209-y), available on [Zenodo](https://zenodo.org/record/4008504#.X4S7RZqxVH4).  We corrupted 1, 5, 10 and 20% of the dataset (N=1,066 images) either by swapping two labels or removing one, and trained with varying percentages of the data.  Then we trained models and evaluated them on the test set (the rest of the data). The figures below show percentage of correct keypoints (PCK) on the test set for various conditions. The results from these experiments can be plotted like this:
```
python CorruptionFigure.py
```

Which creates figures like this (when training with 10% of the data as shown in the paper)

<p align="center">
<img src="ResultsCorruption/LabelingCorruptionimpact_10Percent.png?format=1000w" width="55%">
</p>

Also results for different training fractions (not shown in the paper are plotted).

### Data Augmentation Improves Performance (Figure 8)

Here we used the standad example dataset from the main repo: https://github.com/DeepLabCut/DeepLabCut/tree/master/examples/openfield-Pranav-2018-10-30
and trained with 3 different augmentation methods. To plot the results run the following:

```
python Primer_AugmentationComparison-Figures.py
```
Creates figure (in folder: ResultsComparison)

8A = ResultsComparison/LearningCurves.png

8B = ResultsComparison/ErrorCurves.png

8C, D = Stickfigures.png, Stickfiguresfirst500.png

incl. the 8D in the folder "ResultsComparisonFrames"

We then evaluated this model (only trained on a single mouse and session, which is not recommended) on all other mice and sessions (data from here [Zenodo](https://zenodo.org/record/4008504#.X4S7RZqxVH4)). We noticed that with good augmentation (imgaug, tensorpack) the model generalizes to data from the same camera but not the higher-resolution camera (Fig 8E). We then furthermore illustrate active learning, by adding a few frames from an experiment with the higher-resolution camera (Fig 8F). We found that this is sufficient to generalize reasonably.

```
python Primer_RobustnessEvaluation-Figures.py
```

Creates figure (in folder: ResultsComparison)

8E = ResultsComparison/PCKresults.png

8F = ResultsComparison/PCKresultsafterAugmentation.png

**Outlook**

Do you want to contribute labeled data to the DeepLabCut project?
Check out http://contrib.deeplabcut.org
