import sys
import json
import plotly
import pandas as pd
import joblib

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

from sqlalchemy import create_engine

from fs_ds_util import nlp

HTML_MASTER = 'master.html'

app = Flask(__name__)
context = None  # Initialization near main()


class Context:
    """
    Class used to store information of the application context (DataFrame, Model,
    static graphs, meta data used in other functions).

    An instance of this class will be created after the application is loaded.
    """
    def __init__(self, database_engine, data_frame, model):
        self.database_engine = database_engine
        self.data_frame = data_frame
        self.model = model

        # Required to load WordNet corpus; used by this vectorizer
        nlp.LemmatizingUrlCleaningCountVectorizer(download_nltk_packages=True)

        # Genre data
        self.genre_counts = data_frame.groupby('genre').count()['message']
        self.genre_names = list(self.genre_counts.index)

        # Category data
        self.category_name = data_frame.columns[4:]
        self.category_counts = data_frame[self.category_name].sum().sort_values()

        self.graphs = [
            # Static bar chart (horizontal) representing category distribution
            {
                'data': [
                    Bar(
                        y=self.category_counts.index,
                        x=self.category_counts,
                        orientation='h'
                    )
                ],

                'layout': {
                    'title': 'Distribution of Message Categories',
                    'yaxis': {
                        'title': "Categories"
                    },
                    'xaxis': {
                        'title': "Number of Messages"
                    },
                    'margin': {
                        'l': 200    # required to prevent name cut-off
                    }
                }
            },

            # Static bar chart representing genre distribution
            {
                'data': [
                    Bar(
                        x=self.genre_names,
                        y=self.genre_counts
                    )
                ],

                'layout': {
                    'title': 'Distribution of Message Genres',
                    'yaxis': {
                        'title': "Number of Messages"
                    },
                    'xaxis': {
                        'title': "Genre"
                    }
                }
            }
        ]

        # Encode graphs for plotly.js
        self.html_graph_ids = ["graph-{}".format(i) for i, _ in enumerate(self.graphs)]
        self.graph_json = json.dumps(self.graphs, cls=plotly.utils.PlotlyJSONEncoder)


def load_context():
    """
    Loads the database and the trained model

    :return: Context object containing all the static content of this application
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        engine = create_engine('sqlite:///{}'.format(database_filepath))
        df = pd.read_sql_table('Messages', engine)

        print('Loading model...\n    MODEL: {}'.format(model_filepath))
        model = joblib.load("../data/disaster_message_classifier__RandomForestClassifier.pickle")

        return Context(engine, df, model)
    else:
        print("""
Invalid number of arguments. Please respect the usage message.
        
Usage: python {} DATABASE CLASSIFIER

    DATABASE:   Database containing cleaned messages and categories (.sqlite-file, input).
                Used to display graphs representing the training data.
    CLASSIFIER: Trained classifier used to classify messages (.pickle-file, output).
    
Example: python run.py DisasterResponse.sqlite classifier.pickle
""".format(sys.argv[0]))
        exit(-1)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Request handler of index page.

    This handler takes care of the submitted query and provides data
    for the rendering process of the master page.

    :return: Rendered page template
    """

    # Process the query; if specified in GET request
    query = request.args.get('query', '')
    if query:
        classification_labels = context.model.predict([query])[0]
    else:
        classification_labels = [0]*len(context.category_name)

    classification_results_part1 = dict(zip(context.category_name[:18], classification_labels[:18]))
    classification_results_part2 = dict(zip(context.category_name[18:], classification_labels[18:]))

    # Render web page with necessary data
    return render_template(HTML_MASTER,
                           html_graph_ids=context.html_graph_ids,
                           graph_json=context.graph_json,
                           query=query,
                           classification_result_part1=classification_results_part1,
                           classification_result_part2=classification_results_part2)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


context = load_context()
if __name__ == '__main__':
    main()

