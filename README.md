# Deep Learning Assignment - README
## Overview
This assignment consists of four parts, each implemented as a separate Jupyter Notebook:
1.	Understanding	Neural	Network	Forward	Propagation	and Backpropagation
2.	Predicting Customer Churn using Neural Networks
3.	Spam Classiffication using Recurrent Neural Networks
4.	Image Classiffication using Transfer Learning

Each part includes data preprocessing, model development, training, evaluation, and experimentation with different hyperparameters. The objective is to apply deep learning techniques to various tasks and analyze their performance

## File Structure
•	DL_Assignment_Part1.ipynb

•	DL_Assignment_Part2.ipynb

•	DL_Assignment_Part3.ipynb

•	DL_Assignment_Part4.ipynb

•	DL_Assignment_Report.pdf 

•	README.md

## Setup Instructions
1.	Install Required Dependencies
   
• Ensure you have Python 3.12.7 installed.

• Install the necessary libraries using the following command:
  pip install -r requirements.txt
  
• If requirements.txt is not provided, install manually:
  pip install numpy pandas matplotlib seaborn tensorﬂow keras torch torchvision scikit-learn nltk

3.  Run Jupyter Notebook
   
• Navigate to the directory containing the notebooks and start Jupyter Notebook:


• Open each .ipynb ﬁle and run the cells sequentially.

5.	Dataset Handling

•	For Part 2 and Part 3, datasets need to be downloaded manually from Kaggle:
  Customer Churn Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  Spam	Classiffication	Dataset:	[Spam	Text	Message Classiﬁcation](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)
•	For Part 4 (Image Classiﬁcation), the dataset can be downloaded from:
  [Food Chain Logos](https://www.kaggle.com/datasets/kmkarakaya/logos-bk-kfc-mcdonald-starbucks-subway-none)
• Place the datasets in the appropriate directories before running the notebooks.


## Notes for the Grader

○	Part 1: Implements forward propagation, backpropagation, and gradient descent with both sigmoid and ReLU activations.
○	Part 2: Uses a feedforward neural network for customer churn prediction, with experiments on different architectures.
○	Part 3: Develops an RNN-based spam classiﬁer using LSTM/GRU, achieving high accuracy with optimized hyperparameters.
○	Part 4: Implements image classiﬁcation using VGG-16 with transfer learning and ﬁne-tuning, reaching ~97.40% accuracy.

●	Hyperparameter Tuning: Each model underwent experimentation with different architectures and hyperparameters to improve performance.
●	Potential Issues: If running the image classiﬁcation model (Part 4) on a CPU, training may be slow. Using a GPU is recommended.

## Contact
For any clariﬁcations, please refer to the report (Deep Learning_12320026_Individual Assignment.pdf).

