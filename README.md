# CSE584-Midterm
The following depencies are required to replicate the data and methodology described in the PDF-file in this repository:
  pandas,
  transformers,
  pytorch,
  scikit-learn,
  matplotlib,
  numpy,
  ax-platform

To Generate the Data run generate_data.py: 
  Make sure to download "train.tsv" from https://github.com/google-research-datasets/wiki-split/tree/master and that it is in the same directory as generate_data.py
  
To Train and Evaluate the Model (including generating ROC Curves and saving best model) run train_and_eval.py: 
  Make sure that total_data.csv is in the same directory as train_and_eval.py
