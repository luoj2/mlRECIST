# ml-RECIST project
This contains Python 3 code for the model, figures and select statistics used for a research project titled:

>ml-RECIST: machine learning to estimate RECIST in patients with NSCLC treated with PD-1 blockade.
>Authors: Kathryn C. Arbour1,2*, Luu Anh Tuan3*, Jia Luo1*, Hira Rizvi1, Andrew J. Plodkowski4, Mustafa Sakhi5, Kevin Huang5, Subba R. Digumarthy6, Michelle S. Ginsberg4, Jeffrey Girshman4, Mark G. Kris1,2, Gregory J. Riely1,2, Adam Yala3, Justin F. Gainor4^, Regina Barzilay3^, and Matthew D. Hellmann1,2^
>*Contributed equally, ^Contributed equally 

Author Affiliations: 
1 Thoracic Oncology Service, Memorial Sloan Kettering Cancer Center, New York, NY
2 Department of Medicine, Weill Cornell Medical Center, New York, NY
3 Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute for Technology, Cambridge, MA
4 Department of Radiology, Memorial Sloan Kettering Cancer Center 
5 Department of Medicine, Massachusetts General Hospital, Boston, MA
6 Department of Radiology, Massachusetts General Hospital, Boston, MA

# Code Summary:
Code for model, figures and statistics written for this project include the following:
* ml-RECIST: TensorFlow-based fully connected natural language processing neural network (implementation details found in the manuscript)
* Receiver operator characteristic (ROC) with area under the curve (AUC) estimates
* Survival curves using Kaplan-Meier estimates
* Waterfall plot
* Scatter plot
* Vertical stacked bar plot

# Data Summary:
Data includes the following:
* Reduced dataset of output from ml-RECIST for the training, internal validation and external validation sets

# Installation:
The scripts are dependent on the following packages:
* TensorFlow 1.15
* Gensim
* NumPy
* Matplotlib
* Pandas
* Scikit-learn
* Glove's embeddings glove.840B.300d

Please contact the corresponding authors for any questions or comments regarding the paper.

# Usage:

### Data format: The data input of the models is an Excel file with the columns in the following order:
* Patient ID
* Treatment start date
* Treatment setting
* Outside Scans	(Y/N)
* Objective Response per RECIST (CR/PR/SD/POD))
* Date of radiologic progression-free survival
* PFS censor (0/1)
* Scan timepoint (Baseline/ontx/progression)
* Scan include? (Y/N)
* Date of scan
* Type of scan
* Scan type specified
* Scan report text

### Running code: The model has three main components: predicting the BOR, predicting the FPS(Y/N), and predicting the Progression date.

#### Predicting the BOR: To predict the BOR, run the following command: (see the list of arguments below)

python ./model/src/test_predict_objective.py arguments(optional)

The prediction file is in the folder: log_test/predict_objective

#### Predicting the PFS(Y/N): To predict the PFS(Y/N), run the following command:

python ./model/src/test_predict_progression.py arguments(optional)

The prediction file is in the folder: log_test/predict_progression

#### Predicting the Progression date: To predict the progression date, run the following command:

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

The parameters are tunes among: hidden dimension size (200, 300, 500), dropout (0.8, 0.9, 1.0), learning rate (0.001, 0.005, 0.01), batch_size (1, 2, 4, 8). 


