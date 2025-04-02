# Sample Difficulty Analysis for Computer Audition Tasks
This repository contains and reproduces the results for the paper "Leveraging Sample Difficulty in Computer Audition Analysis", currently under review for EUSIPCO2025.

## Results
We provide the following reproducible results pre-computed from our experiments on the DCASE2020Task1A and the SpeechCommands dataset:
- Overview of training performances under ...;

- Difficulty orderings of samples for both datasets under...;

- Selection of the easiest, medium, and most difficult samples, in raw form, spectrogram and with GradCam explanations ...;

- Graphs appearing in the paper under ...

## Reproduction
The reproduction of the results consists of several steps.

### Environment

Our code depends on several packages, in particular to [autrainer](https://autrainer.github.io/autrainer/) and [aucurriculum](https://autrainer.github.io/aucurriculum/). For further details refer to the referenced documentation. Dependencies can for instance be installed in a virtual environment:

```
cd path/to/repo
python -m venv venv
source venv venv/bin/activate
pip install -r requirements.txt
```

### Base Training

The first step towards the results is a base training of deep learning architectures on the given computer audition tasks based on the aucurriclum (and autrainer) toolkit. The details of the training are defined in `conf/config.yaml` and consist of the [PANNs models](https://github.com/qiuqiangkong/audioset_tagging_cnn) CNN10 and CNN14 models  for the datasets [DCASE2020Task1A](https://dcase.community/challenge2020/task-acoustic-scene-classification) and the [SpeechCommands](https://pytorch.org/audio/main/generated/torchaudio.datasets.SPEECHCOMMANDS.html) dataset with varying random seeds. To download the data, extract Mel-Spectrograms and train the deep learning models run:
```
aucurriculum fetch
aucurriculum preprocess
aucurriculum train
autrainer postprocess results default -a seed model
```

Results of the training are stored in the `resulst/default` directory if not otherly specified.

### Difficulty Scores

The difficulty scores are calculated with the aucurriculum packge. The details of the score calculation is specified in `conf/curriculum.yaml`. To calculate the cumulative accuracy scores for the ensemble of models trained in the previous step, run:

```
aucurriculum curriculum
```

### Evaluation


**Reproduce Inference**
This

First install autrainer version 0.1.0 in a new venv
```
python -m venv venv
source venv/bin/activate
pip install autrainer==0.1.0
```

Unzip the `model.zip` and `inputs_samples_DCASE2020.zip`, the latter into into a directory `inputs`, which may only contain the `.wav` files. 
The layout of the model directory is specific to autrainer as the library has to read a lot of info from the trained model directory to know which model to load.
Finally run
```
autrainer inference model inputs outputs -p log_mel_16k
```

This should create the an `outputs` directory with a `results.csv` file.

autrainer should be installed in `venv/lib/python3.12/site-packages/autrainer` and therein calls the `core/scripts/cli.py`, which ends up calling the final inference code in `serving/serving.py`, where the model prediction happens in the `_predict` method. 
