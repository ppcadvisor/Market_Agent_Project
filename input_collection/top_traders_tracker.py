import os
import requests
import time
import json
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

BULLAWARE_BASE_URL = "https://api.bullaware.com"
API_KEY = os.getenv("BULLAWARE_API_KEY")

class TopTradersTracker:
    """
    Коннектор для BullAware API. Загружает профили из JSON и ищет инвесторов по заданному профилю.
    """
    def __init__(self, profiles_path: str = "config/tracker_profiles.json"):
        if not API_KEY:
            raise ValueError("BULLAWARE_API_KEY не найден в файле .env. Пожалуйста, добавьте его.")
        self.headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Принцип Defensive Programming: Проверяем наличие файла конфигурации
        if not os.path.exists(profiles_path):
            print(f"[Tracker]: WARNING - Файл профилей не найден по пути: {profiles_path}. Трекер не сможет находить инвесторов.")
            self.profiles = {}
        else:
            try:
                with open(profiles_path, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
                print(f"[Tracker]: Успешно загружено {len(self.profiles)} профилей инвесторов.")
            except json.JSONDecodeError as e:
                print(f"[Tracker]: ERROR - Ошибка парсинга JSON в файле {profiles_path}: {e}")
                self.profiles = {}
            except Exception as e:
                print(f"[Tracker]: ERROR - Не удалось загрузить профили из {profiles_path}: {e}")
                self.profiles = {}

    def find_top_investors(self, profile_name: str, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Находит инвесторов по заданному профилю.
        """
        profile = self.profiles.get(profile_name)
        if not profile:
            print(f"[Tracker]: ERROR - Профиль '{profile_name}' не найден в файле конфигурации.")
            return None

        metric = profile.get("ranking_metric", "sharpe-ratio") # Используем sharpe-ratio по умолчанию
        filters = profile.get("api_filters", {})
        filters['limit'] = limit # Добавляем лимит к фильтрам

        url = f"{BULLAWARE_BASE_URL}/v1/rankings/{metric}"
        try:
            print(f"[Tracker]: Запрос инвесторов по профилю '{profile_name}' (метрика: {metric}, фильтры: {filters})...")
            response = requests.get(url, headers=self.headers, params=filters, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "investors" in data and isinstance(data["investors"], list):
                print(f"[Tracker]: Найдено {len(data['investors'])} инвесторов.")
                return data["investors"]
            else:
                print(f"[Tracker]: WARN - Неожиданный формат ответа: {data}")
                return []
        except requests.exceptions.Timeout:
            print(f"[Tracker]: ERROR - Сервер BullAware не ответил за 20 секунд (Timeout).")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[Tracker]: ERROR - Ошибка сети при получении инвесторов: {e}")
            return None

    def get_investor_portfolio(self, username: str) -> Optional[Dict[str, Any]]:
        # Этот метод остается без изменений, он уже достаточно надежен
        url = f"{BULLAWARE_BASE_URL}/v1/investors/{username}/portfolio"
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[Tracker]: ERROR - Не удалось получить портфель для {username}: {e}")
            return None

    def get_social_flow_factor(self, symbol: str, profile_name: str) -> float:
        """
        Главный аналитический метод. Вычисляет "социальный фактор" для одной акции по заданному профилю.
        """
        print(f"\n--- Расчет Social Flow Factor для тикера: {symbol} по профилю '{profile_name}' ---")
        investors_list = self.find_top_investors(profile_name=profile_name)
        
        if investors_list is None:
            print("[Tracker]: Не удалось получить список инвесторов, фактор = 0.0")
            return 0.0
        if not investors_list:
            print("[Tracker]: Список топ-инвесторов по данному профилю пуст, фактор = 0.0")
            return 0.0

        total_investors_checked = len(investors_list)
        holders_count = 0
        
        print(f"Анализирую портфели {total_investors_checked} инвесторов...")
        for i, investor in enumerate(investors_list):
            username = investor.get("username")
            if not username: continue
            
            time.sleep(6)
            print(f"  ({i+1}/{total_investors_checked}) Проверяю портфель {username}...")
            portfolio = self.get_investor_portfolio(username)
            
            if not portfolio or "positions" not in portfolio or not isinstance(portfolio["positions"], list):
                continue
            if any(pos.get("symbol") == symbol for pos in portfolio["positions"]):
                print(f"    -> НАЙДЕНО! {username} держит {symbol}.")
                holders_count += 1
        
        factor = holders_count / total_investors_checked
        print(f"--- РАСЧЕТ ЗАВЕРШЕН: {holders_count} из {total_investors_checked} инвесторов держат {symbol}. Фактор = {factor:.2f} ---")
        return factor

# --- Тестовый блок для проверки ---
if __name__ == '__main__':
    try:
        tracker = TopTradersTracker()
        
        # Тест 1: Проверяем профиль "консервативного инвестора"
        TEST_PROFILE = 'conservative_investor'
        TEST_SYMBOL = 'AAPL'
        print(f"\n{'='*20} ТЕСТ 1: ПРОФИЛЬ '{TEST_PROFILE}' {'='*20}")
        social_factor = tracker.get_social_flow_factor(TEST_SYMBOL, profile_name=TEST_PROFILE)
        print("\n--- Финальный результат теста 1 ---")
        print(f"Итоговый Social Flow Factor для '{TEST_SYMBOL}' (профиль '{TEST_PROFILE}'): {social_factor:.2f}")

        # Тест 2: Проверяем профиль "агрессивного трейдера"
        TEST_PROFILE_2 = 'aggressive_daytrader'
        print(f"\n{'='*20} ТЕСТ 2: ПРОФИЛЬ '{TEST_PROFILE_2}' {'='*20}")
        social_factor_2 = tracker.get_social_flow_factor(TEST_SYMBOL, profile_name=TEST_PROFILE_2)
        print("\n--- Финальный результат теста 2 ---")
        print(f"Итоговый Social Flow Factor для '{TEST_SYMBOL}' (профиль '{TEST_PROFILE_2}'): {social_factor_2:.2f}")

    except ValueError as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА НАСТРОЙКИ: {e}")