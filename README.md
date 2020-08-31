# README

> This is my work for the second project of the Data Science Nanodegree Program from Udacity.

In this project we build a model that predict categories of messages sent during disaster events.

## Disclaimer

I applied the [Udacity Git Commit Message Style Guide](https://udacity.github.io/git-styleguide/) in this repository.

I also used [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/), hence the [Pipfile](https://github.com/mguidoti/DSND-p1-blog/blob/master/Pipfile) and [Pipfile.lock](https://github.com/mguidoti/DSND-p1-blog/blob/master/Pipfile.lock) in this repository.

## Instructions

To run this project, you've to:

- Run ETL pipeline first: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- Then, run the ML pipeline: `python models/train_classifier.py data/DataResponse.db models/classifier.pkl`
- And finally, run the web app: `python run.py`, which will be accessable on http://0.0.0.0:3001.


## Acknowledgements

I often consulted [sklearn documentation](https://scikit-learn.org/stable/index.html), [StackOverflow](https://stackoverflow.com/) (several questions) and these specific GitHub repositories:

- [@shihao-wen](https://github.com/shihao-wen/Udacity-DSND)
- [@nesreensada](https://github.com/nesreensada/Data-Scientist-Udacity-Nanodegree-Term2)
- [@dzakyputra](https://github.com/dzakyputra/udacity-data-scientist-nanodegree)

## License
MIT