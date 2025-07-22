import yfinance as yf
from typing import List, Optional
import pandas as pd

class DataCollector:
    """
    Этот класс отвечает за сбор данных из различных источников.
    Теперь он умеет получать реальные рыночные данные с помощью yfinance.
    """
    def get_price_history(self, symbol: str, period: str = "1mo") -> Optional[pd.Series]:
        """
        Получает исторические цены закрытия для заданного тикера.

        :param symbol: Тикер акции, например "AAPL" для Apple или "MSFT" для Microsoft.
        :param period: Период для загрузки данных ("1d", "5d", "1mo", "3mo", "1y", ...).
        :return: Pandas Series с ценами закрытия или None в случае ошибки.
        """
        try:
            print(f"[Data Collector]: Загрузка данных для {symbol} за период {period}...")
            # Создаем объект "тикер"
            ticker = yf.Ticker(symbol)
            # Запрашиваем историю цен. auto_adjust=True автоматически корректирует цены на сплиты и дивиденды.
            hist = ticker.history(period=period, auto_adjust=True)
            
            if hist.empty:
                print(f"[Data Collector]: Ошибка: не удалось получить данные для {symbol}. Возможно, неверный тикер.")
                return None
            
            print(f"[Data Collector]: Данные успешно загружены.")
            # Возвращаем только столбец с ценами закрытия
            return hist['Close']

        except Exception as e:
            print(f"[Data Collector]: Произошла ошибка при загрузке данных: {e}")
            return None