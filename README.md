# Developing Robust Networks to Defend Against Adversarial Examples

![](/.github/adversarial-examples-visualization.png)

An ongoing area of research involves defenses against adversarial examples, specifically designed inputs that attack the model's inference. We explore the impact on adversarial examples against common deep learning classification architectures such as CNNs and RNNs, specifically testing on a CNN and CNN-LSTM hybrid, trained on the ***CIFAR-10*** dataset. We implemented an efficient adversarial examples attack known as the Fast-Gradient Sign Method (FGSM) to generate perturbations against your baseline model. We then explore the development of robust networks with defenses against adversarial examples by implementing adversarial training by performing data augmentation against the original dataset using FGSM. With a baseline model of 80.374% for the CNN and 79.001% for the CNN-LSTM hybrid architecture and a training $\epsilon = 0.2$, we were able to achieve a validation accuracy of 75.488% for the CNN architecture and 74.056% for the CNN-LSTM hybrid architecture on $\epsilon = 0.1$ adversarial examples resulting in an absolute increase of 64.142% for the CNN architecture and 61.375% for the CNN-LSTM hybrid architecture respectively. When comparing architecture, although there were strengths to the CNN-LSTM hybrid architecture, the CNN outperformed it in terms of accuracy and training time for adversarial examples.

The following repository contains the scripts, models, and data relevant to this project. The data used for this project is from the ***CIFAR-10*** data set. For more information about the implementation and results, please refer to the project paper.

## Repository Structure
- `attacks/`: implementations of adversarial example generation with FGSM
- `data/`: data used for training and testing (included in the .gitignore)
- `models/`: saved checkpoints for the trained models
- `networks/`: implementations of the CNN and CNN-LSTM hybrid architectures
- `utils/`: supporting utility functions
- `analysis.ipynb`: notebook for performing cross architecture comparison
- `cnn.ipynb`: notebook for training and testing the CNN architecture with and without adversarial training
- `rnn.ipynb`: notebook for training and testing the CNN-LSTM hybrid architecture with and without adversarial training
- `main.ipynb`: notebook for an older CNN architecture used to show baseline adversarial example generation
- `(test/train)-cnn.py`: scripts for training and testing the baseline CNN architecture
- `(test/train)-cnn-lstm.py`: scripts for training and testing the baseline CNN-LSTM hybrid architecture

## Installation
In order to run the code and notebooks in this repository, please set up a virtual environment using the following comamnd.

```bash
python3 -m virtualenv venv
```

In order to activate the virtual environment, run the following command.

```bash
source venv/bin/activate
```

Once the virtual environment is activated, you can install all of the necessry dependencies using the following command.

```bash
pip install -r requirements.txt
```

For downloading the dataset, the `torchvision` module should download and create the `data/` containing the training and testing ***CIFAR-10*** dataset when running any of the scripts/notebooks for the first time.

## Authors
This repository was for "Developing Robust Networks to Defend Against Adversarial Examples" by Benson Liu & Isabella Qian for ECE C147: Neural Networks & Deep Learning at UCLA in Winter 2023. For any questions or additional infromation about this project, please contact the authors.