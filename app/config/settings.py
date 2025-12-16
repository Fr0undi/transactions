"""Настройки проекта"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Класс настроек приложения"""

    # Пути
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent  # Корневая директория проекта
    DATA_DIR: Path = BASE_DIR / "data"
    IMAGES_DIR: Path = BASE_DIR / "images"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "analysis.log"

    # Данные
    DATA_FILE: str = "transactions_data.csv"
    SAMPLE_SIZE: int = 100000

    # Модели
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    def __post_init__(self):
        """Создание директорий после инициализации"""

        self.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Глобальный экземпляр настроек
settings = Settings()