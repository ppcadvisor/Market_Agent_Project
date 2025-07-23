import yfinance as yf
from typing import Dict, Any, Optional

class FundamentalParser:
    """
    Этот класс отвечает за сбор и предоставление фундаментальных данных о компании.
    Он использует yfinance для получения ключевых финансовых показателей.
    """
    def get_fundamental_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Получает ключевые фундаментальные показатели для заданного тикера.

        :param symbol: Тикер акции, например "AAPL".
        :return: Словарь с фундаментальными данными или None в случае ошибки.
        """
        try:
            print(f"[Fundamental Parser]: Загрузка фундаментальных данных для {symbol}...")
            ticker = yf.Ticker(symbol)
            
            # .info - это специальный атрибут в yfinance, который содержит огромный
            # словарь с информацией о компании.
            info = ticker.info

            # Defensive Programming: Проверяем, что .info не пустой
            if not info or 'symbol' not in info:
                print(f"[Fundamental Parser]: Ошибка: не удалось получить .info для {symbol}.")
                return None

            # Извлекаем только те данные, которые нас интересуют на данный момент.
            # Используем .get(key, default_value), чтобы избежать ошибок, если какой-то
            # показатель отсутствует для данной акции. Это тоже Defensive Programming.
            fundamental_data = {
                'pe_ratio': info.get('trailingPE', None),          # P/E Ratio (Цена/Прибыль)
                'market_cap': info.get('marketCap', None),        # Рыночная капитализация
                'dividend_yield': info.get('dividendYield', None) # Дивидендная доходность
            }
            
            print(f"[Fundamental Parser]: Данные успешно получены.")
            return fundamental_data

        except Exception as e:
            print(f"[Fundamental Parser]: Произошла непредвиденная ошибка: {e}")
            return None