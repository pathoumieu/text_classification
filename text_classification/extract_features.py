import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from stop_words import get_stop_words


def main(train_df, test_df, output_dir, max_features=10000, n_components=100):
    """
    Extract raw features, tfidf features for train and test, standard scale
    and save them as .csv in output_dir.

    Parameters
    ----------
    train_df: pd.DataFrame
        Dataframe containing Train sentences as `text` column and sequences as
        `sequence` column.
    test_df: pd.DataFrame
        Dataframe containing Test sentences as `text` column and sequences as
        `sequence` column.
    output_dir: str
        Directory in which `Learn/` and `Test/` directories are located.
    max_features: int
        Maximum features in scikit learn TFIDF representation.
    n_components: int
        Number of dimensions in SVD output for LSA.
    """
    train_features = extract_raw_features(train_df)
    test_features = extract_raw_features(test_df)

    train_lsa_features, test_lsa_features = extract_tfidf_features(train_df, test_df,
                                                                   max_features=max_features,
                                                                   n_components=n_components)

    train_features, test_features = concat_and_scale(train_features, train_lsa_features,
                                                     test_features, test_lsa_features)

    train_features.to_csv(output_dir + '/Learn/train_features.csv', index=False)
    test_features.to_csv(output_dir + '/Test/test_features.csv', index=False)


def extract_raw_features(df):
    """
    Extract macro features on sentences.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing sentences as `text` column and sequences as
        `sequence` column.

    Returns
    -------
    features: pd.DataFrame
        Dataframe containing extracted raw features.
    """
    features = pd.DataFrame()

    # Compute sequence features
    features['sequence_length'] = df['sequence'].apply(len)
    features['sequence_unique'] = df['sequence'].apply(lambda x: len(set(x)))

    # Compute raw text features
    features['char_count'] = df['text'].apply(len)
    features['word_count'] = df['text'].apply(lambda x: len(x.split()))
    features['word_density'] = features['char_count'] / (features['word_count'] + 1)
    features['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
    features['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

    return features


def extract_tfidf_features(train_df, test_df, max_features=10000, n_components=100):
    """
    Compute tfidf, n-gram tfidf and n-gram char tfidf features and perform LSA
    on computed features.

    Parameters
    ----------
    train_df: pd.DataFrame
        Dataframe containing Train sentences as `text` column and sequences as
        `sequence` column.
    test_df: pd.DataFrame
        Dataframe containing Test sentences as `text` column and sequences as
        `sequence` column.
    max_features: int
        Maximum features in scikit learn TFIDF representation.
    n_components: int
        Number of dimensions in SVD output for LSA.

    Returns
    -------
    train_lsa_features: pd.DataFrame
        Dataframe containing extracted LSA features for Train.
    test_lsa_features: pd.DataFrame
        Dataframe containing extracted LSA features for Test.
    """
    # Perform word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', max_features=max_features,
                                 stop_words=get_stop_words('french'))
    tfidf_vect.fit(train_df['text'])
    train_tfidf_features = tfidf_vect.transform(train_df['text'])
    test_tfidf_features = tfidf_vect.transform(test_df['text'])

    # Perform ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=max_features)
    tfidf_vect_ngram.fit(train_df['text'])
    train_tfidf_ngram_features = tfidf_vect_ngram.transform(train_df['text'])
    test_tfidf_ngram_features = tfidf_vect_ngram.transform(test_df['text'])

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=max_features)
    tfidf_vect_ngram_chars.fit(train_df['text'])
    train_tfidf_ngram_chars_features = tfidf_vect_ngram_chars.transform(train_df['text'])
    test_tfidf_ngram_chars_features = tfidf_vect_ngram_chars.transform(test_df['text'])

    # Perform LSA
    # Standard tfidf
    svd_tfidf = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd_tfidf.fit(train_tfidf_features)
    train_tfidf_svd_features = svd_tfidf.transform(train_tfidf_features)
    test_tfidf_svd_features = svd_tfidf.transform(test_tfidf_features)

    # 2-3 gram tfidf
    svd_tfidf_ngram = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd_tfidf_ngram.fit(train_tfidf_ngram_features)
    train_tfidf_ngram_svd_features = svd_tfidf_ngram.transform(train_tfidf_ngram_features)
    test_tfidf_ngram_svd_features = svd_tfidf_ngram.transform(test_tfidf_ngram_features)

    # Char 2-3 gram tfidf
    svd_tfidf_ngram_char = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd_tfidf_ngram_char.fit(train_tfidf_ngram_chars_features)
    train_tfidf_ngram_char_svd_features = svd_tfidf_ngram_char.transform(train_tfidf_ngram_chars_features)
    test_tfidf_ngram_char_svd_features = svd_tfidf_ngram_char.transform(test_tfidf_ngram_chars_features)

    train_lsa_features = train_tfidf_svd_features, train_tfidf_ngram_svd_features, train_tfidf_ngram_char_svd_features
    test_lsa_features = test_tfidf_svd_features, test_tfidf_ngram_svd_features, test_tfidf_ngram_char_svd_features

    return train_lsa_features, test_lsa_features


def concat_and_scale(train_features, train_lsa_features, test_features, test_lsa_features):
    """
    Concatenate and standard scale all extracted features for Train and Test.
    """
    train_features = pd.concat([train_features,
                                pd.DataFrame(train_lsa_features[0], index=train_features.index).add_prefix('lsa_tfidf_'),
                                pd.DataFrame(train_lsa_features[1], index=train_features.index).add_prefix('lsa_tfidf_ngram_'),
                                pd.DataFrame(train_lsa_features[2], index=train_features.index).add_prefix('lsa_tfidf_ngram_chars_'),
                               ],
                               axis=1)

    test_features = pd.concat([test_features,
                               pd.DataFrame(test_lsa_features[0], index=test_features.index).add_prefix('lsa_tfidf_'),
                               pd.DataFrame(test_lsa_features[1], index=test_features.index).add_prefix('lsa_tfidf_ngram_'),
                               pd.DataFrame(test_lsa_features[2], index=test_features.index).add_prefix('lsa_tfidf_ngram_chars_'),
                               ],
                               axis=1)

    # Standard scale features
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = pd.DataFrame(scaler.transform(train_features), columns=train_features.columns)
    test_features = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

    return train_features, test_features
