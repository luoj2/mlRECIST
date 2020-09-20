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
>Authors: Kathryn C. Arbour1,2*, Luu Anh Tuan3*, Jia Luo1*, Hira Rizvi1, Andrew J. Plodkowski4, Mustafa Sakhi5, Kevin Huang5, Subba R. Digumarthy6, Michelle S. Ginsberg4, Jeffrey Girshman4, Mark G. Kris1,2, Gregory J. Riely1,2, Adam Yala3, Justin F. Gainor4^, Regina Barzilay3^, and Matthew D. Hellmann1,2^ <i>[accepted, in press] <b>Cancer Discovery</i></b> 2020.

*Contributed equally, ^Contributed equally 

Author Affiliations: 
1 Thoracic Oncology Service, Memorial Sloan Kettering Cancer Center, New York, NY
2 Department of Medicine, Weill Cornell Medical Center, New York, NY
3 Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute for Technology, Cambridge, MA
4 Department of Radiology, Memorial Sloan Kettering Cancer Center 
5 Department of Medicine, Massachusetts General Hospital, Boston, MA
6 Department of Radiology, Massachusetts General Hospital, Boston, MA

# Code Summary:
Code for algorithm, figures and statistics written for this project include the following:
* mlRECIST: TensorFlow-based fully connected natural language processing neural network (implementation details found in the manuscript)
* Receiver operator characteristic (ROC) with area under the curve (AUC) estimates
* Survival curves using Kaplan-Meier estimates
* Waterfall plot
* Scatter plot
* Vertical stacked bar plot

# Data Summary:
Data includes the following:
* Reduced dataset of output from mlRECIST for the training, internal validation, and external validation sets'

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
The data input for the model is an Excel file with the columns in the following order:
* Patient ID, anonymized 
* Treatment start date [MM/DD/YYYY]
* Treatment setting (clinical trial, standard of care)
* Outside Scans	(Y, N)
* Objective Response per RECIST (CR, PR, SD, POD))
* Date of radiologic progression-free survival [MM/DD/YYYY]
* PFS censor (0, 1)
* Scan timepoint (Baseline, ontx, progression)
* Scan include? (Y, N)
* Date of scan [MM/DD/YYYY]
* Type of scan (CT, PET, MR)
* Scan type specified (CT CH/ABD/PEL W/ CON, etc.)
* Scan report text (the entirety of the text report with dates removed)

### Running code:
The model estimates three RECIST outcomes of interest: 
* best overall response (BOR)
* progression (Y, N)
* progression date (MM/DD/YYYY)

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

#### List of arguments and their formats:

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

Developers: [Jia Luo (@luoj2)](https://github.com/luoj2/) and [Anh Tuan Luu (@tuanluu)](https://github.com/tuanluu)

