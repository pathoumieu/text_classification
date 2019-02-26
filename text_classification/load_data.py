import pickle as pkl
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_train(data_dir, features=True, processed_sequences=False):
    """
    Load train dataset, with extracted features if features is True. Encode labels.

    Parameters
    ----------
    data_dir: str
        Directory containing `/Learn/` and `/Test/`.
    features: bool
        If True, load extracted features from text_classification/extract_features.py outputs.
    processed_sequences: bool
        If True, load processed sequences from preprocess_sequences.py outputs.

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing train encoded labels, sentences, sequences, and
        extracted features if features argument is True.
    feature_cols: list
        List of created features names, if features is True.
    """
    with open(data_dir + '/Learn/labels.pkl', 'rb') as f:
        labels = pkl.load(f)

    with open(data_dir + '/Learn/sentences.pkl', 'rb') as f:
        sentences = pkl.load(f)

    if processed_sequences:
        with open(data_dir + '/Learn/processed_sequences.pkl', 'rb') as f:
            sequences = pkl.load(f)

    else:
        with open(data_dir + '/Learn/sequences.pkl', 'rb') as f:
            sequences = pkl.load(f)

    df = pd.DataFrame()
    df['text'] = sentences
    df['sequence'] = sequences
    df['labels'] = labels

    # Encode labels in binary
    le = LabelEncoder()
    df.labels = le.fit_transform(df.labels)

    if features:
        features = pd.read_csv(data_dir + '/Learn/train_features.csv')
        df = pd.concat([df, features], axis=1)
        feature_cols = features.columns
        return df, feature_cols

    else:
        return df


def load_test(data_dir, features=True, processed_sequences=False):
    """
    Load test dataset, with extracted features if features is True.
    Parameters
    ----------
    data_dir: str
        Directory containing `/Learn/` and `/Test/`.
    features: bool
        If True, load extracted features from text_classification/extract_features.py outputs.
    processed_sequences: bool
        If True, load processed sequences from preprocess_sequences.py outputs.

    Returns
    -------
    df: pd.DataFrame
        Dataframe containing test sentences, sequences, and extracted features
        if features argument is True.
    feature_cols: list
        List of created features names, if features is True.
    """
    with open(data_dir + '/Test/sentences.pkl', 'rb') as f:
        sentences = pkl.load(f)

    if processed_sequences:
        with open(data_dir + '/Test/processed_sequences.pkl', 'rb') as f:
            sequences = pkl.load(f)

    else:
        with open(data_dir + '/Test/sequences.pkl', 'rb') as f:
            sequences = pkl.load(f)

    df = pd.DataFrame()
    df['text'] = sentences
    df['sequence'] = sequences

    if features:
        features = pd.read_csv(data_dir + '/Test/test_features.csv')
        df = pd.concat([df, features], axis=1)
        feature_cols = features.columns
        return df, feature_cols

    else:
        return df
