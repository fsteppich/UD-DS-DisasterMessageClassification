import sys
import time
import pickle
import util     # local util file


import pandas as pd
import numpy as np
import sqlalchemy as sql

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')    # download for lemmatization

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from fs_ds_util import nlp
from fs_ds_util import sysout as out


def load_data(database_filepath):
    """
    Load data from database file into DataFrame

    :param database_filepath: sqlite-file containing cleaned messages and assigned categories
    :return X: pandas.DataFrame containing the util.COL_MESSAGE
    :return Y: pandas.DataFrame containing the assigned categories of each message
    :return category_names: List of categories in DataFrame Y
    """
    engine = sql.create_engine('sqlite:///{}'.format(database_filepath))

    if engine.has_table(table_name=util.DB_TABLE_MESSAGES):
        out.info('Loading data from table \'{}\''.format(util.DB_TABLE_MESSAGES))
    else:
        message = 'Table \'{}\' not found in sqlite database {}'.format(util.DB_TABLE_MESSAGES, database_filepath)
        out.error(message)
        raise ValueError(message)

    with engine.connect() as conn:
        conn.begin()
        df = pd.read_sql_table(util.DB_TABLE_MESSAGES, conn)
        conn.close()

    out.info("Table loaded into DataFrame with {} columns and {} rows".format(df.shape[1], df.shape[0]))

    category_names = df.columns.tolist()[4:]  # skip first 4 (id, ..., genre) columns
    # category_names = list({'water', 'medical_help', 'medical_products'})
    out.info("Using {} columns as classification target: {}".format(len(category_names), category_names))

    X = df[util.COL_MESSAGE]
    Y = df[category_names]
    return X, Y, category_names


def build_model(grid_search=False):
    """
    Build the data model / pipeline

    :param grid_search: Specify if grid search should be used to find optimal parameters
    :return: The data model
    """
    pipeline = Pipeline([
        ('vect', nlp.LemmatizingUrlCleaningCountVectorizer()),
        #('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        #('clf', MultiOutputClassifier(RandomForestClassifier()))
        #('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])

    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2))
        #        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_df': (.5, 1.0)  # (.5, .75, 1.0)
        #        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #        'features__text_pipeline__tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': (50, 100, 200)
        #        'clf__min_samples_split': [2, 3, 4],
        #        'features__transformer_weights': (
        #            {'text_pipeline': 1, 'starting_verb': 0.5},
        #            {'text_pipeline': 0.5, 'starting_verb': 1},
        #            {'text_pipeline': 0.8, 'starting_verb': 1},
        #        )
    }

    if grid_search:
        pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1, cv=2)

    return pipeline


def train_model(model, X_train, Y_train):
    """
    Train data model on given training data set.

    :param model: Model to be trained
    :param X_train: Input data set of model for training
    :param Y_train: Output data set of model for training
    :return: The trained model
    """
    out.info("Training model on {} message(s)".format(len(X_train)))
    model.fit(X_train, Y_train)

    training_results = model.cv_results_
    rank_test_scores = training_results['rank_test_score'].tolist()
    idx_best_ranked = rank_test_scores.index(min(rank_test_scores))

    out.info("Best Parameters: {} @ {} (mean test score)".format(
        training_results['params'][idx_best_ranked],
        training_results['mean_test_score'][idx_best_ranked]
    ))

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the trained model on the given test data set.

    Prints scores on each category and averaged to stdout.

    :param model: Model to be evaluated
    :param X_test: Input data set of model for evaluation / test
    :param Y_test: Output data set of model for evaluation / test
    :param category_names: List of categories available in the given model
    :return: Evaluated model
    """
    out.info("Evaluating model on {} message(s)".format(len(X_test)))
    Y_pred = model.predict(X_test)

    total_f1 = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    out.warn("Suppressing zero division warnings in classification report. Using 1.0 if score is ill-defined.\n"
             "    Watch for 1.0 scores and decide if they appear to often.")
    for idx, category in enumerate(category_names):
        report = classification_report(Y_test[category], Y_pred[:, idx], zero_division=1, output_dict=True)

        precision = report['weighted avg']['precision']
        total_precision += precision

        recall = report['weighted avg']['recall']
        total_recall += recall

        f1 = report['weighted avg']['f1-score']
        total_f1 += f1

        accuracy = report['accuracy']
        total_accuracy += accuracy

        out.info("{:25} F1: {:5.3}   Precision: {:5.3}   Recall: {:5.3}   Accuracy: {:5.3}".format(
            category, f1, precision, recall, accuracy
        ))

    print()
    number_of_categories = len(category_names)
    out.info("{:25} F1: {:5.3}   Precision: {:5.3}   Recall: {:5.3}   Accuracy: {:5.3}".format(
        ">> Avg.", total_f1/number_of_categories, total_precision/number_of_categories,
        total_recall/number_of_categories, total_accuracy/number_of_categories
    ))

    return model


def save_model(model, model_filepath):
    """
    Store model into pickle database

    :param model: Model to be serialized into pickle file
    :param model_filepath: File name of the pickle file
    :return: None
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        file.close()



def main():
    if len(sys.argv) == 3:
        start_time_program = time.time()
        database_filepath, model_filepath = sys.argv[1:]
        print()

        start_time = time.time()
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Building model...')
        model = build_model(grid_search=True)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Training model...')
        model = train_model(model, X_train, Y_train)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Evaluating model...')
        model = evaluate_model(model, X_test, Y_test, category_names)
        util.print_elapsed_time(start_time, time.time())

        start_time = time.time()
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        util.print_elapsed_time(start_time, time.time())

        print()
        print('Trained model saved!')
        util.print_elapsed_time(start_time_program, time.time(), prompt="Total execution time")

    else:
        print("""
Invalid number of arguments. Please respect the usage message.
        
Usage: python {} DATABASE CLASSIFIER

    DATABASE:   Database containing cleaned messages and categories (.sqlite-file, input)
    CLASSIFIER: File to store the trained classifier (.pickle-file, output)
    
Example: python train_classifier.py DisasterResponse.sqlite classifier.pickle
""".format(sys.argv[0]))


if __name__ == '__main__':
    main()
