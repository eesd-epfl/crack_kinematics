# Crack Kinematics
This repository contains the codes for computing crack kinematics using a binary mask that represents a segmented crack. The methodoly hereby implementes was presented in the paper ["Determing crack kinematics from imaged crack patterns"](https://doi.org/10.1016/j.conbuildmat.2022.128054) by Pantoja-Rosero et., al.

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
