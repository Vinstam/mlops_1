import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def clean_url_series(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'https?://', ' ', regex=True)
    s = s.str.replace(r'www\.', ' ', regex=True)
    s = s.str.replace(r'utm_[a-z_]+=[^&\s]+', ' ', regex=True)
    s = s.str.replace(r'[^0-9a-zа-яё]+', ' ', regex=True)
    return s


def normalize_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace('ё', 'е', regex=False)
    s = s.str.replace(r'[^0-9a-zа-яе]+', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def make_text(df: pd.DataFrame) -> pd.Series:
    name = normalize_text_series(df['name'])
    desc = normalize_text_series(df['description'])
    model = normalize_text_series(df['model'])
    type_prefix = normalize_text_series(df['type_prefix'])
    vendor = normalize_text_series(df['vendor'])
    url = clean_url_series(df['url'])
    image_url = clean_url_series(df['image_url'])

    text = (
        (' name ' + name) * 3 +
        (' type ' + type_prefix) * 3 +
        (' model ' + model) * 2 +
        (' vendor ' + vendor) * 2 +
        ' desc ' + desc +
        ' url ' + url +
        ' image ' + image_url
    )

    return text.str.replace(r'\s+', ' ', regex=True).str.strip()


def make_ids_for_submission(test_df: pd.DataFrame) -> np.ndarray:
    for col in ['ID', 'id']:
        if col in test_df.columns:
            return test_df[col].to_numpy()

    return np.arange(0, len(test_df))


def prepare_data(train_df, test_df=None, valid_size=None, random_state=42):
    X_train_text = make_text(train_df)
    X_test_text = make_text(test_df) if test_df is not None else None

    y_full = train_df['category_ind'].astype(int).to_numpy()

    result = {
        'X_train_text': X_train_text,
        'X_test_text': X_test_text,
        'y_full': y_full,
    }

    if valid_size is not None:
        counts = pd.Series(y_full).value_counts()

        can_stratify = np.array([
            counts[label] >= 2 for label in y_full
        ])

        idx_common = np.where(can_stratify)[0]
        idx_rare = np.where(~can_stratify)[0]

        train_common_idx, valid_idx = train_test_split(
            idx_common,
            test_size=valid_size,
            random_state=random_state,
            stratify=y_full[idx_common],
        )

        train_idx = np.concatenate([train_common_idx, idx_rare])

        result['train_idx'] = train_idx
        result['valid_idx'] = valid_idx

    return result


def prepare_tfidf_features(
    X_train_text,
    X_valid_text=None,
    X_test_text=None,
    word_max_features=180_000,
    char_max_features=120_000,
):
    result = {}

    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents=None,
        max_features=word_max_features,
        dtype=np.float32,
    )

    X_train_word = word_vectorizer.fit_transform(X_train_text)

    result['word_vectorizer'] = word_vectorizer
    result['X_train_word'] = X_train_word

    if X_valid_text is not None:
        result['X_valid_word'] = word_vectorizer.transform(X_valid_text)

    if X_test_text is not None:
        result['X_test_word'] = word_vectorizer.transform(X_test_text)

    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=char_max_features,
        dtype=np.float32,
    )

    X_train_char = char_vectorizer.fit_transform(X_train_text)

    result['char_vectorizer'] = char_vectorizer
    result['X_train_char'] = X_train_char

    if X_valid_text is not None:
        result['X_valid_char'] = char_vectorizer.transform(X_valid_text)

    if X_test_text is not None:
        result['X_test_char'] = char_vectorizer.transform(X_test_text)

    result['X_train'] = hstack(
        [X_train_word, X_train_char],
        format='csr',
        dtype=np.float32,
    )


    if X_valid_text is not None:
        result['X_valid'] = hstack(
            [result['X_valid_word'], result['X_valid_char']],
            format='csr',
            dtype=np.float32,
        )

    if X_test_text is not None:
        result['X_test'] = hstack(
            [result['X_test_word'], result['X_test_char']],
            format='csr',
            dtype=np.float32,
        )
    return result
