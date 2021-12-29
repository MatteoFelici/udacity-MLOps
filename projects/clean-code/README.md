# Predict Customer Churn
Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Given a Jupyter notebook with code to develop an end-to-end Churn model, refactor the code following all best practises learned during the course. In particular:
- write clean, modular code, with docstrings and loggers
- develop a test suite


## Running Files
All requirements to run the code are into `requirements.txt`. To create a suitable Python environment with `conda`, run

```bash
conda create -n your-env-name python=3.7
conda activate your-env-name
pip install -r requirements.txt
```

The main file to develop the model is `churn_library.py`:
```python
python churn_library.py --input-path path/to/input-data.csv
```
There are additional constants that can be edited into `constants.py`, like the input features list and the hyperparameters grid to use with the `GridSearchCV` object.

Finally, to run the test suite, simply run
```bash
pytest
```
in your main directory.
