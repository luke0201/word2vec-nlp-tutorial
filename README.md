# Bag of Words Meets Bags of Popcorn

This is my Kaggle submission code.

## Requirements

You need Python 3.6 or higher version.

The following libraries are required.

- pandas
- scikit-learn
- nltk

## Usage

Download the dataset using the following command.

```
kaggle competitions download -c word2vec-nlp-tutorial
unzip unlabeledTrainData.tsv.zip && rm unlabeledTrainData.tsv.zip
unzip testData.tsv.zip && rm testData.tsv.zip
```

Then run `word2vec-nlp-tutorial.py` as follows.

```
python word2vec-nlp-tutorial.py
```

Finally, submit the result using the following command. Replace the `<submission_message>` by yours.

```
kaggle competitions submit -c word2vec-nlp-tutorial -f submission.csv -m <submission_message>
```
