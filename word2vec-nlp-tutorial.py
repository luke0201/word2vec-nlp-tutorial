from argparse import ArgumentParser
import csv
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
import nltk

nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_data(path):
    df = pd.read_csv(path, sep='\t', quoting=3)

    return df


def transform(X):
    X.drop('id', axis=1, inplace=True)

    X['review'] = X['review'].str.replace('<br />', ' ', regex=False)
    X['review'] = X['review'].str.replace(r'[^A-Za-z]', ' ', regex=True)

    return X


def predict(X):
    sentiment_intensity_analyzer = SentimentIntensityAnalyzer()

    def vader_polarity(review, threshold=0.1):
        scores = sentiment_intensity_analyzer.polarity_scores(review)
        compound_score = scores['compound']
        sentiment = 1 if compound_score >= threshold else 0
        return sentiment

    predictions = X['review'].apply(vader_polarity)
    return predictions


def parse_args():
    parser = ArgumentParser(
        description='Generate the submission file for Kaggle House Prices competition.')
    parser.add_argument(
        '--test', type=Path, default='testData.tsv',
        help='path of testData.tsv downloaded from the competition')

    return parser.parse_args()


def main(args):
    X_test = load_data(args.test)
    ids = X_test['id']
    X_test = transform(X_test)
    y_test = predict(X_test)

    submission = {
        'id': ids,
        'sentiment': y_test
    }
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index=False, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    sys.exit(main(parse_args()))
