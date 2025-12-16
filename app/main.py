"""
================================================================================
Анализ транзакций по банковским картам
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (classification_report, confusion_matrix,
                            mean_squared_error, r2_score, accuracy_score,
                            silhouette_score)
from sklearn.neural_network import MLPClassifier

from app.config.settings import settings
from app.utils.logger import setup_logger

warnings.filterwarnings('ignore')


logger = setup_logger(__name__)


# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


# ================================================================================
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ================================================================================

def load_and_preprocess_data(filepath: str, sample_size: int = 100000) -> pd.DataFrame:
    """Загрузка и предобработка данных транзакций"""

    logger.info("=" * 80)
    logger.info("1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ")
    logger.info("=" * 80)

    # Загрузка данных
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, nrows=sample_size)
        logger.info(f"Загружено {sample_size:,} строк (выборка из большого файла)")
    else:
        df = pd.read_excel(filepath, nrows=sample_size)

    logger.info(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
    logger.info(f"Столбцы датасета: {list(df.columns)}")

    # Информация о данных
    logger.info("--- Информация о типах данных ---")
    logger.debug(f"\n{df.dtypes}")

    logger.info("--- Первые 5 строк ---")
    logger.debug(f"\n{df.head()}")

    # Проверка пропусков
    logger.info("--- Пропуски в данных ---")
    missing = df.isnull().sum()
    missing_info = missing[missing > 0]
    if len(missing_info) > 0:
        logger.info(f"Найдены пропуски:\n{missing_info}")
    else:
        logger.info("Пропуски отсутствуют")

    # Обработка столбца amount
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False)
        df['amount'] = df['amount'].str.replace(',', '', regex=False)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Преобразование даты
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek

    # Удаление дубликатов
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    logger.info("--- Удаление дубликатов ---")
    logger.info(f"Найдено дубликатов: {duplicates_before}")
    logger.info(f"Размер после удаления: {df.shape[0]} строк")

    # Заполнение пропусков
    logger.info("--- Заполнение пропусков ---")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"  {col}: заполнено медианой ({median_val})")
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"  {col}: заполнено модой ({mode_val})")

    # Удаление аномалий
    if 'amount' in df.columns:
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df['amount'] < lower_bound) | (df['amount'] > upper_bound)).sum()
        logger.info("--- Обработка выбросов (IQR метод) ---")
        logger.info(f"Нижняя граница: {lower_bound:.2f}")
        logger.info(f"Верхняя граница: {upper_bound:.2f}")
        logger.info(f"Количество выбросов: {outliers}")

        df['amount_cleaned'] = df['amount'].clip(lower=lower_bound, upper=upper_bound)

    # Создание целевой переменной
    if 'amount' in df.columns:
        df['is_large_transaction'] = (df['amount'].abs() > 100).astype(int)

    # Создание категории MCC
    if 'mcc' in df.columns:
        df['mcc_category'] = pd.cut(df['mcc'],
                                     bins=[0, 2000, 4000, 6000, 8000, 10000],
                                     labels=['Услуги', 'Транспорт', 'Розница', 'Развлечения', 'Прочее'])

    logger.info("--- Итоговый размер данных ---")
    logger.info(f"Строк: {df.shape[0]}, Столбцов: {df.shape[1]}")

    logger.debug("--- Описательная статистика ---")
    logger.debug(f"\n{df.describe()}")

    return df


# ================================================================================
# 2. ВИЗУАЛИЗАЦИЯ ДАННЫХ
# ================================================================================

def visualize_data(df: pd.DataFrame) -> None:
    """Визуализация данных"""

    logger.info("=" * 80)
    logger.info("2. ВИЗУАЛИЗАЦИЯ ДАННЫХ")
    logger.info("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    if 'amount' in df.columns:
        axes[0, 0].hist(df['amount'].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_title('Распределение суммы транзакций')
        axes[0, 0].set_xlabel('Сумма ($)')
        axes[0, 0].set_ylabel('Частота')

    if 'dayofweek' in df.columns:
        days = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        day_counts = df['dayofweek'].value_counts().sort_index()
        axes[0, 1].bar(days, day_counts.values, color='coral', edgecolor='black')
        axes[0, 1].set_title('Количество транзакций по дням недели')
        axes[0, 1].set_xlabel('День недели')
        axes[0, 1].set_ylabel('Количество')

    if 'hour' in df.columns:
        hour_counts = df['hour'].value_counts().sort_index()
        axes[0, 2].plot(hour_counts.index, hour_counts.values, marker='o', color='green', linewidth=2)
        axes[0, 2].fill_between(hour_counts.index, hour_counts.values, alpha=0.3, color='green')
        axes[0, 2].set_title('Распределение транзакций по часам')
        axes[0, 2].set_xlabel('Час')
        axes[0, 2].set_ylabel('Количество')

    if 'use_chip' in df.columns:
        chip_counts = df['use_chip'].value_counts()
        axes[1, 0].pie(chip_counts.values, labels=chip_counts.index, autopct='%1.1f%%',
                       colors=plt.cm.Pastel1.colors, startangle=90)
        axes[1, 0].set_title('Типы транзакций')

    if 'merchant_state' in df.columns:
        state_counts = df['merchant_state'].value_counts().head(10)
        axes[1, 1].barh(state_counts.index, state_counts.values, color='mediumpurple', edgecolor='black')
        axes[1, 1].set_title('Топ-10 штатов по транзакциям')
        axes[1, 1].set_xlabel('Количество')
        axes[1, 1].invert_yaxis()

    if 'amount_cleaned' in df.columns and 'use_chip' in df.columns:
        df_sample = df.sample(min(10000, len(df)))
        df_sample.boxplot(column='amount_cleaned', by='use_chip', ax=axes[1, 2])
        axes[1, 2].set_title('Сумма транзакций по типу')
        axes[1, 2].set_xlabel('Тип транзакции')
        axes[1, 2].set_ylabel('Сумма ($)')
        plt.suptitle('')

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")


# ================================================================================
# 3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ================================================================================

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Корреляционный анализ"""

    logger.info("=" * 80)
    logger.info("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    logger.info("=" * 80)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['id', 'client_id', 'card_id', 'merchant_id']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    logger.info(f"Числовые переменные для анализа: {numeric_cols}")

    corr_matrix = df[numeric_cols].corr()

    logger.info("--- Корреляционная матрица ---")
    logger.debug(f"\n{corr_matrix.round(3)}")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, square=True)
    plt.title('Корреляционная матрица', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'correlation_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    logger.info("--- Наиболее значимые корреляции ---")
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append({
                'Переменная 1': corr_matrix.columns[i],
                'Переменная 2': corr_matrix.columns[j],
                'Корреляция': corr_matrix.iloc[i, j]
            })

    corr_df = pd.DataFrame(corr_pairs)
    corr_df['Абс. корреляция'] = corr_df['Корреляция'].abs()
    corr_df = corr_df.sort_values('Абс. корреляция', ascending=False)
    logger.debug(f"\n{corr_df.head(10).to_string(index=False)}")

    return corr_matrix


# ================================================================================
# 4. РЕГРЕССИОННЫЙ АНАЛИЗ
# ================================================================================

def regression_analysis(df: pd.DataFrame) -> LinearRegression:
    """Регрессионный анализ"""

    logger.info("=" * 80)
    logger.info("4. РЕГРЕССИОННЫЙ АНАЛИЗ")
    logger.info("=" * 80)

    features = ['hour', 'dayofweek', 'month']
    available_features = [f for f in features if f in df.columns]

    if 'amount_cleaned' not in df.columns or len(available_features) == 0:
        logger.warning("Недостаточно данных для регрессионного анализа")
        return None

    df_reg = df[available_features + ['amount_cleaned']].dropna()
    X = df_reg[available_features]
    y = df_reg['amount_cleaned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    logger.info(f"Размер обучающей выборки: {len(X_train)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")

    logger.info("--- 4.1 Простая линейная регрессия (hour -> amount) ---")

    X_simple = X_train[['hour']]
    X_simple_test = X_test[['hour']]

    lr_simple = LinearRegression()
    lr_simple.fit(X_simple, y_train)
    y_pred_simple = lr_simple.predict(X_simple_test)

    logger.info(f"Коэффициент (b1): {lr_simple.coef_[0]:.4f}")
    logger.info(f"Свободный член (b0): {lr_simple.intercept_:.4f}")
    logger.info(f"R2 score: {r2_score(y_test, y_pred_simple):.4f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_simple)):.4f}")

    logger.info("--- 4.2 Множественная линейная регрессия ---")
    logger.info(f"Признаки: {available_features}")

    lr_multiple = LinearRegression()
    lr_multiple.fit(X_train, y_train)
    y_pred_multiple = lr_multiple.predict(X_test)

    logger.info("Коэффициенты модели:")
    for feature, coef in zip(available_features, lr_multiple.coef_):
        logger.info(f"  {feature}: {coef:.4f}")
    logger.info(f"Свободный член: {lr_multiple.intercept_:.4f}")

    r2_multiple = r2_score(y_test, y_pred_multiple)
    rmse_multiple = np.sqrt(mean_squared_error(y_test, y_pred_multiple))

    logger.info(f"R2 score: {r2_multiple:.4f}")
    logger.info(f"RMSE: {rmse_multiple:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X_simple_test, y_test, alpha=0.3, s=10, label='Фактические')
    axes[0].plot(X_simple_test.sort_values('hour'),
                 lr_simple.predict(X_simple_test.sort_values('hour')),
                 color='red', linewidth=2, label='Регрессия')
    axes[0].set_xlabel('Час')
    axes[0].set_ylabel('Сумма транзакции ($)')
    axes[0].set_title('Простая линейная регрессия')
    axes[0].legend()

    axes[1].scatter(y_test, y_pred_multiple, alpha=0.3, s=10)
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label='Идеальное предсказание')
    axes[1].set_xlabel('Фактические значения')
    axes[1].set_ylabel('Предсказанные значения')
    axes[1].set_title(f'Множественная регрессия (R2 = {r2_multiple:.4f})')
    axes[1].legend()

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'regression_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    return lr_multiple


# ================================================================================
# 5. КЛАССИФИКАЦИЯ: ДЕРЕВЬЯ РЕШЕНИЙ
# ================================================================================

def decision_tree_classification(df: pd.DataFrame) -> DecisionTreeClassifier:
    """Классификация с помощью дерева решений"""

    logger.info("=" * 80)
    logger.info("5. КЛАССИФИКАЦИЯ: ДЕРЕВЬЯ РЕШЕНИЙ")
    logger.info("=" * 80)

    features = ['hour', 'dayofweek', 'month', 'mcc']
    available_features = [f for f in features if f in df.columns]

    if 'is_large_transaction' not in df.columns or len(available_features) == 0:
        logger.warning("Недостаточно данных для классификации")
        return None

    df_clf = df[available_features + ['is_large_transaction']].dropna()
    X = df_clf[available_features]
    y = df_clf['is_large_transaction']

    logger.info(f"Признаки: {available_features}")
    logger.info(f"Целевая переменная: is_large_transaction (0 - обычная, 1 - крупная)")
    logger.info(f"Распределение классов:\n{y.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    dt_classifier = DecisionTreeClassifier(
        max_depth=5, min_samples_split=100, random_state=settings.RANDOM_STATE
    )
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info("--- Результаты классификации ---")
    logger.info(f"Accuracy: {accuracy:.4f}")

    logger.info("Отчет классификации:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Обычная', 'Крупная'])}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Матрица ошибок:\n{cm}")

    logger.info("Важность признаков:")
    for feature, importance in zip(available_features, dt_classifier.feature_importances_):
        logger.info(f"  {feature}: {importance:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_tree(dt_classifier, feature_names=available_features,
              class_names=['Обычная', 'Крупная'], filled=True,
              rounded=True, ax=axes[0], fontsize=8, max_depth=3)
    axes[0].set_title('Дерево решений (глубина 3)')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Обычная', 'Крупная'],
                yticklabels=['Обычная', 'Крупная'])
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_title('Матрица ошибок')

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'decision_tree.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    return dt_classifier


# ================================================================================
# 6. КЛАССИФИКАЦИЯ: АЛГОРИТМ KNN
# ================================================================================

def knn_classification(df: pd.DataFrame) -> tuple:
    """Классификация методом K ближайших соседей"""

    logger.info("=" * 80)
    logger.info("6. КЛАССИФИКАЦИЯ: АЛГОРИТМ KNN")
    logger.info("=" * 80)

    features = ['hour', 'dayofweek', 'month', 'mcc']
    available_features = [f for f in features if f in df.columns]

    if 'is_large_transaction' not in df.columns or len(available_features) == 0:
        logger.warning("Недостаточно данных для классификации")
        return None

    df_clf = df[available_features + ['is_large_transaction']].dropna()

    if len(df_clf) > 50000:
        df_clf = df_clf.sample(50000, random_state=settings.RANDOM_STATE)

    X = df_clf[available_features]
    y = df_clf['is_large_transaction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    logger.info(f"Признаки: {available_features}")
    logger.info(f"Размер обучающей выборки: {len(X_train)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")

    logger.info("--- Подбор оптимального K ---")
    k_range = range(1, 21)
    accuracies = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        accuracies.append(knn.score(X_test, y_test))

    optimal_k = k_range[np.argmax(accuracies)]
    logger.info(f"Оптимальное K: {optimal_k}")
    logger.info(f"Лучшая точность: {max(accuracies):.4f}")

    knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_optimal.fit(X_train, y_train)
    y_pred = knn_optimal.predict(X_test)

    logger.info(f"--- Результаты классификации (K={optimal_k}) ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    logger.info("Отчет классификации:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Обычная', 'Крупная'])}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Матрица ошибок:\n{cm}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, accuracies, marker='o', linewidth=2, markersize=6)
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное K={optimal_k}')
    axes[0].set_xlabel('Количество соседей (K)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Зависимость точности от K')
    axes[0].legend()
    axes[0].grid(True)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Обычная', 'Крупная'],
                yticklabels=['Обычная', 'Крупная'])
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_title(f'Матрица ошибок KNN (K={optimal_k})')

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'knn_classification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    return knn_optimal, scaler


# ================================================================================
# 7. КЛАСТЕРНЫЙ АНАЛИЗ: K-MEANS
# ================================================================================

def kmeans_clustering(df: pd.DataFrame) -> tuple:
    """Кластерный анализ методом K-Means"""

    logger.info("=" * 80)
    logger.info("7. КЛАСТЕРНЫЙ АНАЛИЗ: K-MEANS")
    logger.info("=" * 80)

    features = ['amount_cleaned', 'hour', 'dayofweek']
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        logger.warning("Недостаточно данных для кластеризации")
        return None

    df_cluster = df[available_features].dropna()

    if len(df_cluster) > 30000:
        df_cluster = df_cluster.sample(30000, random_state=settings.RANDOM_STATE)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)

    logger.info(f"Признаки для кластеризации: {available_features}")
    logger.info(f"Размер выборки: {len(df_cluster)}")

    logger.info("--- Метод локтя ---")
    inertias = []
    silhouettes = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=settings.RANDOM_STATE, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

    optimal_k = K_range[np.argmax(silhouettes)]
    logger.info(f"Оптимальное K по силуэту: {optimal_k}")

    kmeans_final = KMeans(n_clusters=optimal_k, random_state=settings.RANDOM_STATE, n_init=10)
    clusters = kmeans_final.fit_predict(X_scaled)

    df_cluster['cluster'] = clusters

    logger.info(f"--- Характеристики кластеров (K={optimal_k}) ---")
    cluster_stats = df_cluster.groupby('cluster')[available_features].mean()
    logger.info(f"\n{cluster_stats.round(2)}")

    logger.info("Размеры кластеров:")
    logger.info(f"\n{df_cluster['cluster'].value_counts().sort_index().to_string()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].plot(K_range, inertias, marker='o', linewidth=2)
    axes[0, 0].set_xlabel('Количество кластеров (K)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Метод локтя')
    axes[0, 0].grid(True)

    axes[0, 1].plot(K_range, silhouettes, marker='s', linewidth=2, color='green')
    axes[0, 1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Оптимальное K={optimal_k}')
    axes[0, 1].set_xlabel('Количество кластеров (K)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Коэффициент силуэта')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    scatter = axes[1, 0].scatter(df_cluster[available_features[0]],
                                  df_cluster[available_features[1]],
                                  c=clusters, cmap='viridis', alpha=0.5, s=10)
    axes[1, 0].set_xlabel(available_features[0])
    axes[1, 0].set_ylabel(available_features[1])
    axes[1, 0].set_title(f'Кластеры (K={optimal_k})')
    plt.colorbar(scatter, ax=axes[1, 0], label='Кластер')

    cluster_counts = df_cluster['cluster'].value_counts().sort_index()
    axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='coral', edgecolor='black')
    axes[1, 1].set_xlabel('Кластер')
    axes[1, 1].set_ylabel('Количество наблюдений')
    axes[1, 1].set_title('Размеры кластеров')

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'kmeans_clustering.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    return kmeans_final, scaler


# ================================================================================
# 8. НЕЙРОННЫЕ СЕТИ
# ================================================================================

def neural_network_classification(df: pd.DataFrame) -> tuple:
    """Классификация с помощью нейронной сети"""

    logger.info("=" * 80)
    logger.info("8. НЕЙРОННЫЕ СЕТИ (MLP CLASSIFIER)")
    logger.info("=" * 80)

    features = ['hour', 'dayofweek', 'month', 'mcc']
    available_features = [f for f in features if f in df.columns]

    if 'is_large_transaction' not in df.columns or len(available_features) == 0:
        logger.warning("Недостаточно данных для классификации")
        return None

    df_nn = df[available_features + ['is_large_transaction']].dropna()

    if len(df_nn) > 50000:
        df_nn = df_nn.sample(50000, random_state=settings.RANDOM_STATE)

    X = df_nn[available_features]
    y = df_nn['is_large_transaction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    logger.info(f"Признаки: {available_features}")
    logger.info(f"Размер обучающей выборки: {len(X_train)}")
    logger.info(f"Размер тестовой выборки: {len(X_test)}")

    logger.info("--- Архитектура сети ---")
    logger.info(f"Входной слой: {len(available_features)} нейронов")
    logger.info("Скрытый слой 1: 64 нейрона (ReLU)")
    logger.info("Скрытый слой 2: 32 нейрона (ReLU)")
    logger.info("Выходной слой: 2 класса (Softmax)")

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=settings.RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )

    logger.info("Обучение модели...")
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info("--- Результаты классификации ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Количество итераций: {mlp.n_iter_}")
    if mlp.best_loss_ is not None:
        logger.info(f"Лучшая функция потерь: {mlp.best_loss_:.4f}")
    else:
        logger.info(f"Финальная функция потерь: {mlp.loss_curve_[-1]:.4f}")

    logger.info("Отчет классификации:")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Обычная', 'Крупная'])}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Матрица ошибок:\n{cm}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(mlp.loss_curve_, linewidth=2)
    axes[0].set_xlabel('Итерация')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Кривая обучения нейронной сети')
    axes[0].grid(True)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
                xticklabels=['Обычная', 'Крупная'],
                yticklabels=['Обычная', 'Крупная'])
    axes[1].set_xlabel('Предсказанный класс')
    axes[1].set_ylabel('Истинный класс')
    axes[1].set_title(f'Матрица ошибок нейронной сети (Accuracy={accuracy:.4f})')

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'neural_network.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")

    return mlp, scaler


# ================================================================================
# 9. СРАВНЕНИЕ МОДЕЛЕЙ
# ================================================================================

def compare_models(df: pd.DataFrame) -> None:
    """Сравнение всех моделей классификации"""

    logger.info("=" * 80)
    logger.info("9. СРАВНЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
    logger.info("=" * 80)

    features = ['hour', 'dayofweek', 'month', 'mcc']
    available_features = [f for f in features if f in df.columns]

    if 'is_large_transaction' not in df.columns or len(available_features) == 0:
        return

    df_clf = df[available_features + ['is_large_transaction']].dropna()
    if len(df_clf) > 30000:
        df_clf = df_clf.sample(30000, random_state=settings.RANDOM_STATE)

    X = df_clf[available_features]
    y = df_clf['is_large_transaction']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE
    )

    results = {}

    dt = DecisionTreeClassifier(max_depth=5, random_state=settings.RANDOM_STATE)
    dt.fit(X_train, y_train)
    results['Decision Tree'] = accuracy_score(y_test, dt.predict(X_test))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    results['KNN (K=5)'] = accuracy_score(y_test, knn.predict(X_test))

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=settings.RANDOM_STATE)
    mlp.fit(X_train, y_train)
    results['Neural Network'] = accuracy_score(y_test, mlp.predict(X_test))

    logger.info("--- Сравнение точности моделей ---")
    for model, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {model}: {acc:.4f}")

    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = list(results.values())
    colors = ['steelblue', 'coral', 'mediumpurple']

    bars = plt.bar(models, accuracies, color=colors, edgecolor='black')
    plt.ylabel('Accuracy')
    plt.title('Сравнение моделей классификации')
    plt.ylim(0, 1)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    save_path = settings.IMAGES_DIR / 'model_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"График сохранен: {save_path}")


# ================================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ================================================================================

def main():
    """Главная функция"""

    logger.info("=" * 80)
    logger.info("АНАЛИЗ ТРАНЗАКЦИЙ ПО БАНКОВСКИМ КАРТАМ")
    logger.info("=" * 80)

    logger.info(f"Директория для графиков: {settings.IMAGES_DIR}")
    logger.info(f"Директория для логов: {settings.LOGS_DIR}")
    logger.info(f"Файл логов: {settings.LOG_FILE}")

    # Путь к файлу данных
    filepath = settings.DATA_FILE

    try:
        df = load_and_preprocess_data(filepath, sample_size=settings.SAMPLE_SIZE)
        visualize_data(df)
        correlation_analysis(df)
        regression_analysis(df)
        decision_tree_classification(df)
        knn_classification(df)
        kmeans_clustering(df)
        neural_network_classification(df)
        compare_models(df)

        logger.info("=" * 80)
        logger.info("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
        logger.info("=" * 80)
        logger.info(f"Графики сохранены в: {settings.IMAGES_DIR}")
        logger.info(f"Логи сохранены в: {settings.LOG_FILE}")

    except FileNotFoundError:
        logger.error(f"Файл '{filepath}' не найден!")
        logger.error("Убедитесь, что файл находится в текущей директории.")
    except Exception as e:
        logger.exception(f"Произошла ошибка: {str(e)}")
        raise


if __name__ == "__main__":
    main()