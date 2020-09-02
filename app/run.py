import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM Messages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data for second chart
    disaster_categories = (df[['genre', 'search_and_rescue',
        'infrastructure_related', 'weather_related']]
        .groupby('genre').sum())

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                # changed first chart for a pie chart
                Pie(
                    labels = genre_names,
                    values = genre_counts,
                    marker = {
                        'colors': [
                            '#DBDBDB',
                            '#808080',
                            '#383838',
                        ]
                    },
                    sort = False
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                # add a second chart, bar chart, with the counts per
                # message related type and genre
                Bar(
                    x = list(disaster_categories.index),
                    y = disaster_categories['search_and_rescue'],
                    name = 'Search & Rescue',
                    marker_color = ['#DBDBDB', '#DBDBDB', '#DBDBDB']

                ),
                Bar(
                    x = list(disaster_categories.index),
                    y = disaster_categories['infrastructure_related'],
                    name = 'Infrastructured',
                    marker_color = ['#808080', '#808080', '#808080']
                ),
                Bar(

                    x = list(disaster_categories.index),
                    y = disaster_categories['weather_related'],
                    name = 'Weather',
                    marker_color = ['#383838', '#383838', '#383838']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Related Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()