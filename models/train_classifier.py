import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine

# Import natural language library and specific packages
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import stuff from sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Import Grid Search
from sklearn.model_selection import GridSearchCV

# Import estimators
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

nltk.download(["punkt", "wordnet"])


def load_data(database_filepath):
    """Loads a sqlite database, returning the input and output varibles

    Args:
        database_filepath (string): Filepath to a sqlite database

    Returns:
        numpy.ndarray: X, set of input variables 
        numpy.ndarray: y, set of output variables
        numpy.ndarray: category_names, list of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)

    df = pd.read_sql('SELECT * FROM Messages', engine)

    df2 = df.head(100)

    X = df2.message.values
    y = df2.drop(columns=["id", "message", "original", "genre"])
    category_names = y.columns[4:]

    return X, y, category_names


def tokenize(text):
    """Reduce the words passed as parameter to their root form (stemming), and
    matched them to their roots (lemmatization), returning the converted tokens

    Args:
        text (string): Message to be tokenized

    Returns:
        list: List of cleaned, tokenized words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens


def build_model():
    """Build the model with Grid Search

    Returns:
        sklearn.model_selection: Model with the best parameter values possible
            for the provided selection of parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # Commented this off because this classifier was taking too long to run
        #('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
    'vect__max_df': [0.5, 0.7],
    'tfidf__use_idf': [True, False],
    # Commented this off because it's a KNeighborsClassifier parameter
    #'clf__estimator__weights': ["uniform", "distance"],
    'clf__estimator__n_estimators': [25, 50, 60]
    }

    grid_search = GridSearchCV(pipeline, param_grid=parameters)

    return grid_search


def evaluate_model(model, X_test, y_test, category_names):
    """Predict data based on the test set of the input variables

    Args:
        model (sklearn.model_selection): Model
        X_test (numpy.ndarray): Test set of the input variables
        y_test (pandas.DataFrame): Test set of the output variables
        category_names (numpy.ndarray): List of category names
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        if i > 0:
            print('column {}, index {}: '.format(col, i))
            print(classification_report(y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """Save the model in a pickle file determined by the user

    Args:
        model (sklearn.model_selection): Model
        model_filepath (string): Filepath to save the pickle file
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()