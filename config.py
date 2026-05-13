import os
from sklearn.linear_model import SGDClassifier
from pathlib import Path


EXPERIMENT_NAME = "[JUNK] ECOM-product-classifiers"
PROJECT_NAME = "production-ml-spring-2026"
##### в какой директории будет лежать проект
BASE_DIR = Path("/Users/nsdemidov/Projects/education_projects/VK-EDUCATION/mlops/project")
#####
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.parquet.snappy"
TEST_PATH = DATA_DIR / "test.parquet.snappy"
TREE_PATH = DATA_DIR / "tree.csv"
ZIP_PATH = DATA_DIR / f"{PROJECT_NAME}.zip"


SUBMISSION_PATH = BASE_DIR / "submission.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

MLRUNS_PATH = BASE_DIR / "mlruns"

RANDOM_STATE = 42
VALID_SIZE = 0.20


os.environ['KAGGLE_USERNAME'] = 'YourKaggleName'
os.environ['KAGGLE_KEY'] = 'YourKaggleApiToken'

MODELS_CONFIG = [
    {
        'name': 'SGDClassifier',
        'model': SGDClassifier(
            loss='log_loss', penalty='l2', alpha=0.0001,
            max_iter=10, tol=1e-3, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=-1
        )
    },
  ### остальные модели - что хотелось попробовать, но не хватило компьюта
    # {
    #     'name': 'CatBoost',
    #     'model': CatBoostClassifier(
    #         iterations=10, learning_rate=0.1, depth=6,
    #         loss_function='MultiClass', random_state=RANDOM_STATE,
    #         thread_count=-1, verbose=False, auto_class_weights='Balanced'
    #     )
    # },
    # {
    #     'name': 'RandomForest',
    #     'model': RandomForestClassifier(
    #         n_estimators=10, class_weight='balanced',
    #         random_state=RANDOM_STATE, n_jobs=-1
    #     )
    # },
    # {
    #     'name': 'MLPClassifier',
    #     'model': MLPClassifier(
    #         hidden_layer_sizes=(64, 64), activation='relu', solver='adam',
    #         alpha=0.001, batch_size=64, learning_rate='adaptive',
    #         learning_rate_init=0.001, max_iter=2,
    #         early_stopping=True, n_iter_no_change=10, random_state=RANDOM_STATE
    #     )
    # }
]
