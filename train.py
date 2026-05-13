import time
import warnings

import pandas as pd
import joblib

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

from catboost import CatBoostClassifier
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi


import mlflow
import mlflow.sklearn

from config import (
    EXPERIMENT_NAME,
    TRAIN_PATH,
    DATA_DIR,
    TRAIN_PATH,
    TEST_PATH,
    TREE_PATH,
    ZIP_PATH,
    TEST_PATH,
    PROJECT_NAME,
    SUBMISSION_PATH,
    MODEL_PATH,
    MLRUNS_PATH,
    RANDOM_STATE,
    VALID_SIZE,
    MODELS_CONFIG
)

from data_processing import (
    prepare_data,
    prepare_tfidf_features,
    make_ids_for_submission,
)

warnings.filterwarnings('ignore')

def download_data_if_needed():
    required_files = [
        TRAIN_PATH,
        TEST_PATH,
        TREE_PATH,
    ]

    if all(path.exists() for path in required_files):
        print("Данные уже существуют, скачивание не требуется")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files(
        PROJECT_NAME,
        path=DATA_DIR,
        force=True,
    )

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Данные успешно скачаны и распакованы")



def fit_and_eval_model(model, model_name, X_train, y_train, X_valid, y_valid):
    print(f"=== Начало обучения {model_name} ===")

    start_time = time.time()

    model.fit(X_train, y_train)
    valid_pred = model.predict(X_valid)

    train_time = time.time() - start_time

    macro_f1 = f1_score(y_valid, valid_pred, average='macro')
    weighted_f1 = f1_score(y_valid, valid_pred, average='weighted')

    print(f"Время обучения: {train_time:.1f} сек")
    print(f"=== Конец обучения {model_name} ===")

    metric_result = {
        'model_name': model_name,
        'model': model,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'train_time': train_time,
        'predictions': valid_pred,
    }

    return model, metric_result


def get_model_config(model_name, models_config):
    for model_config in models_config:
        if model_config['name'] == model_name:
            return model_config['model']

    raise ValueError(f'Модель {model_name} не найдена')


def make_predict(model, X_test, test_df, output_path):
    test_pred = model.predict(X_test).astype(int)

    submission = pd.DataFrame({
        'ID': make_ids_for_submission(test_df),
        'category_ind': test_pred,
    })

    submission.to_csv(output_path, index=False)
    print(f'Предсказание успешно сохранено в {output_path}')

    return test_pred


def run_validation(train, models_config):
    data = prepare_data(
        train,
        valid_size=VALID_SIZE,
        random_state=RANDOM_STATE,
    )

    features = prepare_tfidf_features(
        data['X_train_text'],
        X_valid_text=data['X_train_text'].iloc[data['valid_idx']],
        X_test_text=None,
    )

    mlflow.set_tracking_uri(f"file:{MLRUNS_PATH}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for model_config in models_config:
        with mlflow.start_run(run_name=model_config['name']):
            mlflow.log_params(model_config['model'].get_params())

            best_model, result = fit_and_eval_model(
                model_config['model'],
                model_config['name'],
                features['X_train'][data['train_idx']],
                data['y_full'][data['train_idx']],
                features['X_valid'],
                data['y_full'][data['valid_idx']],
            )

            mlflow.log_metrics({
                "macro_f1": result['macro_f1'],
                "weighted_f1": result['weighted_f1'],
                "train_time": result['train_time'],
            })

            mlflow.sklearn.log_model(best_model, name="model")

            results.append(result)

    print("=" * 20 + "СРАВНЕНИЕ МОДЕЛЕЙ:" + "=" * 20)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('macro_f1', ascending=False)

    print(results_df[['model_name', 'macro_f1', 'weighted_f1', 'train_time']])

    best_model_result = results_df.iloc[0]

    return best_model_result, results_df


def train_final_and_predict(train, test, best_model_result, models_config):
    data_full = prepare_data(
        train,
        test_df=test,
        valid_size=None,
    )

    features_full = prepare_tfidf_features(
        data_full['X_train_text'],
        X_test_text=data_full['X_test_text'],
    )

    X_train = features_full['X_train']

    best_model = get_model_config(
        best_model_result['model_name'],
        models_config,
    )

    best_model.fit(X_train, data_full['y_full'])

    test_pred = make_predict(
        best_model,
        features_full['X_test'],
        test,
        SUBMISSION_PATH,
    )

    joblib.dump(
        {
            'word_vectorizer': features_full['word_vectorizer'],
            'char_vectorizer': features_full['char_vectorizer'],
            'classifier': best_model,
            'macro_f1_validation': float(best_model_result['macro_f1']),
            'feature_columns': [
                'name',
                'description',
                'model',
                'type_prefix',
                'vendor',
                'url',
                'image_url',
            ],
        },
        MODEL_PATH,
    )

    print(f'Модель сохранена в {MODEL_PATH}')

    return test_pred


def main():
    download_data_if_needed()

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    best_model_result, _ = run_validation(
        train,
        MODELS_CONFIG,
    )

    train_final_and_predict(
        train,
        test,
        best_model_result,
        MODELS_CONFIG,
    )


if __name__ == "__main__":
    main()
