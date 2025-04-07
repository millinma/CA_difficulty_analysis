# Sample Difficulty Analysis for Computer Audition Tasks
This repository contains and reproduces the results for the paper "Leveraging Sample Difficulty in Computer Audition Analysis", currently under review for EUSIPCO2025.

## Results
We provide the following reproducible results pre-computed from our experiments on the DCASE2020Task1A and the SpeechCommands dataset:
- Overview of training under `results/experiments/summary/`, as well as `results/experiments/agg_seed_model/`;

- Sample difficulty ordering under `results/experiments/curriculum/CumulativeAccuracy/CumulativeAccuracy[dataset].csv`;

- Selection of the easiest, medium, and most difficult samples, in raw form, spectrogram and with GradCam explanations, as well as graphs for the paper under `results/analysis/` (Note that samples from DCASE2020Task1A need to be downloaded individually due to the corresponding license);

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
autrainer fetch
autrainer preprocess
autrainer train
autrainer postprocess results default -a seed model
```

Results of the training are stored in the `results/default` directory if not otherly specified.

### Difficulty Scores

The difficulty scores are calculated with the aucurriculum packge. The details of the score calculation is specified in `conf/curriculum.yaml`. To calculate the cumulative accuracy scores for the ensemble of models trained in the previous step, run:

```
aucurriculum curriculum
```

### Analysis
To reproduce the analysis reported in the paper, run the following three scripts in order. If necessary, adjust filepaths in the `util_functions.py`

```
python class_analysis.py # performs class-level analysis
python analyse_difficulty_spectrum.py # calculates/saves raw data, spectrograms and explanations of the extreme difficulty ends
python postprocess_results.py # creates the plots reported in the paper
```