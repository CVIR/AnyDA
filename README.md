# AnyDA: Anytime Domain Adaptation [[Paper]](https://openreview.net/pdf?id=yyLvxYBJV1B) [[Website]](https://cvir.github.io/projects/anyda)

This repository contains the implementation details of our ICLR 2023 paper **AnyDA: Anytime Domain Adaptation**.

Omprakash Chakraborty*, Aadarsh Sahoo, Rameswar Panda and Abir Das, "AnyDA: Anytime Domain Adaptation"\

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{chakraborty2022anyda,
  title={{AnyDA: Anytime Domain Adaptation}},
  author={Chakraborty, Omprakash and Sahoo, Aadarsh and Panda, Rameswar and Das, Abir},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```

### Preparing the Environment

#### Conda 
Please use the `anyda.yml` file to create the conda environment `anyda` as:

`conda env create -f anyda.yml`

### Data Directory Structure
All the datasets should be stored in the folder `./data` following the convention `./data/<dataset_name>/<domain_names>`. E.g. for `Office31` the structure would be as follows:

    .
    ├── ...
    ├── data
    │   ├── Office31
    │   │    ├── amazon
    │   │    ├── webcam
    │   │    ├── dslr
    │   └── ...
    └── ...

For using datasets stored in some other directories, please update the path to the data accordingly in the txt files inside the folder `./data_labels`.

The official download links for the datasets used for this paper are:

**Office31**: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code

**OfficeHome**: http://hemanthdv.org/OfficeHome-Dataset/

**DomainNet**: http://ai.bu.edu/M3SDA/

### Hyperparameters
All the training hyperparameters for a given dataset should be saved as `.yml` file and stored in the folder `./apps`. A sample file has been provided in the `./apps` folder name `resnet50_office31.yml` 

### Training AnyDA
Here is a sample and recomended command to train AnyDA for the transfer task of `Amazon -> Webcam` from `Office31` dataset:

`CUDA_VISIBLE_DEVICES=0 python AnyDA.py app:apps/resnet50_office31.yml`
