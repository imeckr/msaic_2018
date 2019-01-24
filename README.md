# Microsoft AI Challenge 2018 - Solution
My solution to Microsoft AI Challenge 2018

This solution achieved a spot in **Top 20 teams out of 2000 teams** that were participating\[\*]. 

Microsoft organized *Microsoft AI Challenge 2018* for India in November 2018. The problem statement was to figure out the right response to queries(web) given 10 possible responses. 

## How to use?
- Create a python(3.6) enviroment with libraries from `requirements.txt`
- Place the dataset in `data` folder i.e. it should contain `Data.tsv` and `eval1_unlabelled.tsv` as provided by the organisers
- Run preprocessing `python preprocess.py`(It standardizes the column names, creates a local validation dataset and undersamples the training dataset for fast and effective training. All the processed files will be saved in `processed` folder)
- Run the training `python training.py`, it saves the model parameters in `model_weights` folder as well
- The above steps also generates the submission in the current directory

## Files
- `dynamic_clip_attention.py` - Model
- `utils.py` - Utility Functions
- `preprocess.py` - Preprocessing
- `training.py` - Training and prediction

## References
This solution is a modified version of https://github.com/wjbianjason/Dynamic-Clip-Attention

\[\*] The actual solution was an ensemble of models.
