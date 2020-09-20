# mlRECIST project

<!-- Information badges -->
<p align="center">
    <a href="https://www.repostatus.org/#active">
    <img alt="Repo status" src="https://www.repostatus.org/badges/latest/inactive.svg" />
  </a>
  <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/dedalus">
  <img alt="PyPI - License" src="https://img.shields.io/pypi/l/dedalus">
</p>

mlRECIST is a machine learning classification algorithm (deep natural language processing [NLP]) we developed that estimates Response Evaluation Criteria in Solid Tumors (RECIST) outcomes from radiology text reports. This model is NOT intended to replicate RECIST or be used in clinical practice or trials but rather is a tool for analysis of retrospective data. 

This repository contains our open-source Python code for the model, example of the output (reduced data), and select statistical/ plotting files. We are unable to share the input data because it is protected health information (PHI). For details please see our manuscript:

>Deep learning to estimate RECIST in patients with NSCLC treated with PD-1 blockade. 
>Kathryn C. Arbour*<sup>1,2</sup>, Luu Anh Tuan*<sup>3</sup>, Jia Luo*<sup>1</sup>, Hira Rizvi<sup>1</sup>, Andrew J. Plodkowski<sup>4</sup>, Mustafa Sakhi<sup>5</sup>, Kevin Huang<sup>5</sup>, Subba R. Digumarthy<sup>6</sup>, Michelle S. Ginsberg<sup>4</sup>, Jeffrey Girshman<sup>4</sup>, Mark G. Kris<sup>1,2</sup>, Gregory J. Riely<sup>1,2</sup>, Adam Yala<sup>3</sup>, Justin F. Gainor^<sup>5</sup>, Regina Barzilay^<sup>3</sup>, and Matthew D. Hellmann^<sup>1,2</sup> <i>[accepted, in press] <b>Cancer Discovery</i></b> 2020.

*Contributed equally, ^Contributed equally 

Author Affiliations: 
1 Thoracic Oncology Service, Memorial Sloan Kettering Cancer Center, New York, NY
2 Department of Medicine, Weill Cornell Medical Center, New York, NY
3 Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute for Technology, Cambridge, MA
4 Department of Radiology, Memorial Sloan Kettering Cancer Center 
5 Department of Medicine, Massachusetts General Hospital, Boston, MA
6 Department of Radiology, Massachusetts General Hospital, Boston, MA

# Code Summary:
Code for the algorithm, figures and statistics written for this project include the following:
* mlRECIST: TensorFlow-based fully connected natural language processing neural network (implementation details found in the manuscript)
* Receiver operator characteristic (ROC) with area under the curve (AUC) estimates
* Survival curves using Kaplan-Meier estimates
* Waterfall plot
* Scatter plot
* Vertical stacked bar plot

# Data Summary:
Data includes the following:
* Reduced dataset of output from mlRECIST for the training, internal validation, and external validation sets

# Installation:
The scripts are dependent on the following packages:
* TensorFlow 1.15
* Gensim
* NumPy
* Matplotlib
* Pandas
* Scikit-learn
* Glove's embeddings glove.840B.300d

# Usage:

### Data format: 
The data used (including input, ground truth, and accompanying information) was formatted as an Excel file with these columns in this exact order:
* Patient ID, anonymized 
* Treatment start date [MM/DD/YYYY]
* Treatment setting (clinical trial, standard of care)
* Outside Scans	(Y, N)
* Objective Response per RECIST (CR, PR, SD, POD)
* Date of radiologic progression-free survival [MM/DD/YYYY]
* PFS censor (0, 1)
* Scan timepoint (Baseline, ontx, progression)
* Scan include? (Y, N)
* Date of scan [MM/DD/YYYY]
* Type of scan (CT, PET, MR)
* Scan type specified (CT CH/ABD/PEL W/ CON, etc.)
* Scan report text (the entirety of the text report)

Ultimately, the input for the algorithm is column:
* Scan report text

The model estimates three RECIST outcomes of interest: 
* best overall response (BOR) (CR, PR, SD, POD)
* progression (Y, N)
* progression date (MM/DD/YYYY)

Specifically these are the following columns that served as ground truth:
* Objective Response per RECIST
* Date of radiologic progression-free survival
* PFS censor

### Running code:

#### Predicting BOR: 
To predict BOR, run the following command: (see the list of arguments below)

    python ./model/src/test_predict_objective.py arguments(optional)

The prediction file is in the folder: log_test/predict_objective

#### Predicting progression (Y, N): 
To predict progression (Y, N), run the following command:

    python ./model/src/test_predict_progression.py arguments(optional)

The prediction file is in the folder: log_test/predict_progression

#### Predicting progression date: 
To predict the progression date, run the following command:

    python ./model/src/test_predict_date.py arguments(optional)

The prediction file is in the folder: log_test/predict_date

#### List of arguments and their format:

Path files:
* Training data: --data_source path_to_training_data
* Testing data: --data_test path_to_testing_data
* Embedding file: --embedding path_to_embedding_file

Parameters of models:
* Embedding size: --embedding_size value 
* Hidden dimension size: --hidden_dim value
* Dropout: --dropout_keep_prob value
* Learning rate: --learning_rate value
* Batch size_ --batch_size value
* Number of training  epochs: --num_epochs value

The parameters can be tuned among: hidden dimension size (200, 300, 500), dropout (0.8, 0.9, 1.0), learning rate (0.001, 0.005, 0.01), and batch_size (1, 2, 4, 8). 

Please contact the corresponding authors Regina Barzilay or Matthew D. Hellmann for any questions or comments regarding the paper.

Authors: [Jia Luo (@luoj2)](https://github.com/luoj2/) and [Anh Tuan Luu (@tuanluu)](https://github.com/tuanluu)

