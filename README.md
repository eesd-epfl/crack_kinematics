# Crack Kinematics (tested on ubuntu 18.04 lts)
This repository contains the codes for computing crack kinematics using a binary mask that represents a segmented crack. The methodoly hereby implementes was presented in the paper ["Determing crack kinematics from imaged crack patterns"](https://doi.org/10.1016/j.conbuildmat.2022.128054) by Pantoja-Rosero et., al.


<p align="center">
  <img src=docs/images/ck_01.png>
</p>


<p align="center">
  <img src=docs/images/ck_02.png>
</p>


<p align="center">
  <img src=docs/images/ck_03.png>
</p>


<p align="center">
  <img src=docs/images/ck_04.png>
</p>


<p align="center">
  <img src=docs/images/ck_05.png>
</p>


<p align="center">
  <img src=docs/images/ck_06.png>
</p>


<p align="center">
  <img src=docs/images/ck_07.png>
</p>


<p align="center">
  <img src=docs/images/ck_08.png>
</p>


<p align="center">
  <img src=docs/images/ck_03.png>
</p>


<p align="center">
  <img src=docs/images/ck_01.png>
</p>


## How to use it?

### 1. Clone repository

Clone repository in your local machine. All codes related with method are inside the `src` directory.

### 2. Download data

Download data file  from [Data](https://doi.org/10.5281/zenodo.6632071). Extract the folder `data/` and place it inside the repository folder

#### 2a. Repository directory

The repository directory should look as:

```
dt_smw
└───src
└───data
└───results
```

### 3. Environment

Create a conda environment and install python packages. At the terminal in the repository location.

`conda create -n crack_kinematics python=3.8`

`conda activate crack_kinematics`

`pip install -r requirements.txt`

### 4. Using method

The main functions of the methodology to compute the crack kinematics of crack patterns using binary mask images are placed in `src/least_square_crack_kinematics.py`. Test them with the examples provided as follows:

`python example/example_kinematics_pattern.py`
`python example/example_kinematics_patch.py`

### 5. Using your own data

The methodology requires as input a binary mask that represents a segmented crack. Create a folder containing the image to be analysed inside the data folder and run the algorithms as shown in the example files `src/example_kinematics_pattern.py` or `src/example_kinematics_patch.py`.

### 6. Results

The results will be saved inside `results` folder with the same name of the folder containing the input image. This contain a json file with all the displacements computed for the crack skeleton. Further, figures that represent the kinematics are output in the same folder.

### 7. Paper experiments

The scripts used to run the experiments presented in the paper can be found inside the folder `paper_examples`

### 8. Citation

We kindly ask you to cite us if you use this project, dataset or article as reference.

Paper:
```
@article{Pantoja-Rosero2020c,
title = {Determining crack kinematics from imaged crack patterns},
journal = {Construction and Building Materials},
volume = {343},
pages = {128054},
year = {2022},
issn = {0950-0618},
doi = {https://doi.org/10.1016/j.conbuildmat.2022.128054},
url = {https://www.sciencedirect.com/science/article/pii/S0950061822017202},
author = {B.G. Pantoja-Rosero and K.R.M. {dos Santos} and R. Achanta and A. Rezaie and K. Beyer},
}
```
Dataset:
```
@dataset{Pantoja-Rosero2020c-ds,
  author       = {Pantoja-Rosero Bryan German and
                  Dos Santos Ketson and
                  Achanta Radhakrishna and
                  Rezaie Amir and
                  Beyer Katrin},
  title        = {{Dataset for determining crack kinematics from 
                   imaged crack patterns}},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v0.0},
  doi          = {10.5281/zenodo.6632071},
  url          = {https://doi.org/10.5281/zenodo.6632071}
}
```
