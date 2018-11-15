# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Libraries used in the project
Pandas, Numpy, Matplotlib, Seaborn, pickle, sklearn, re, nltk, sqlalchemy,
flask, json, plotly

### Motivation for the project
The project is part of a Udacity Nanodegree in Data Science. In the project I
set up an ETL pipeline to clean and store data in a database, and set up a
ML pipeline to train, classify and save.

### Files in the repository
process_data.py - python script for loading, cleaning and saving data
README.md - This file, providing some documentation for the project
disaster_messages.csv, disaster_categories.csv - csv files to be processed
train_classifier.py - ML pipeline that trains and saves classifier
run.py - runs the web app
