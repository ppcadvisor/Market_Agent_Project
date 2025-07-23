import pandas as pd
from typing import List, Optional, Dict, Any

class TechnicalAnalyzer:
    """
    Этот класс отвечает за вычисление технических индикаторов на основе рыночных данных.
    Теперь он умеет считать RSI и MACD.
    """

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """
        Вычисляет Индекс Относительной Силы (RSI).
        (Логика этого метода не изменилась)
        """
        if len(prices) < period + 1:
            return None

        price_series = pd.Series(prices)
        delta = price_series.diff(1)
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        if avg_loss.iloc[-1] == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi.iloc[-1], 2)


    # --- НОВЫЙ МЕТОД ---
    def calculate_macd(self, prices: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Optional[Dict[str, Any]]:
        """
        Вычисляет MACD (Схождение/расхождение скользящих средних) и определяет сигнал пересечения.

        :param prices: Список цен закрытия.
        :param fast_period: Период для быстрой EMA.
        :param slow_period: Период для медленной EMA.
        :param signal_period: Период для сигнальной линии EMA.
        :return: Словарь с состоянием MACD или None, если данных недостаточно.
        """
        if len(prices) < slow_period:
            return None

        price_series = pd.Series(prices)

        # 1. Рассчитываем быструю и медленную экспоненциальные скользящие средние (EMA)
        ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()

        # 2. Рассчитываем линию MACD (разница между быстрой и медленной EMA)
        macd_line = ema_fast - ema_slow

        # 3. Рассчитываем сигнальную линию (EMA от самой линии MACD)
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # --- Логика определения пересечения (самое важное) ---
        # Мы смотрим на две последние точки во времени: "вчера" (индекс -2) и "сегодня" (индекс -1)
        
        # Если "вчера" линия MACD была НИЖЕ сигнальной, а "сегодня" стала ВЫШЕ,
        # это "бычий" кроссовер (сигнал к покупке).
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            crossover = "BULLISH_CROSSOVER"
        
        # Если "вчера" линия MACD была ВЫШЕ сигнальной, а "сегодня" стала НИЖЕ,
        # это "медвежий" кроссовер (сигнал к продаже).
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            crossover = "BEARISH_CROSSOVER"
            
        # Если пересечения не было
        else:
            crossover = "NO_CROSSOVER"

        # Возвращаем словарь с полной информацией
        return {
            'macd_line': round(macd_line.iloc[-1], 2),
            'signal_line': round(signal_line.iloc[-1], 2),
            'crossover': crossover
        }
        
        # --- ДОБАВЬТЕ ЭТОТ КОД В КОНЕЦ ФАЙЛА technical_analyzer.py ---

# --- Демонстрация работы класса ---
if __name__ == '__main__':
    # Создаем экземпляр нашего анализатора
    analyzer = TechnicalAnalyzer()
    
    # --- Тест 1: Имитируем "медвежий" сценарий (сильный рост, потом начало падения) ---
    print("--- Тест 1: Медвежий сценарий ---")
    bearish_prices = [
        150, 152, 155, 158, 161, 164, 166, 167, 168, 169, # Сильный рост
        168.5, 167, 165, 163, 161 # Начало падения
    ]
    print(f"Цены: {bearish_prices}")
    
    # Вызываем наши методы для расчета
    rsi_value = analyzer.calculate_rsi(bearish_prices, period=14)
    macd_result = analyzer.calculate_macd(bearish_prices)
    
    print(f"Результат RSI: {rsi_value}")
    print(f"Результат MACD: {macd_result}")
    
    # --- Тест 2: Имитируем "бычий" сценарий (сильное падение, потом начало роста) ---
    print("\n--- Тест 2: Бычий сценарий ---")
    bullish_prices = [
        169, 167, 165, 163, 160, 157, 155, 154, 153, 152, # Сильное падение
        152.5, 154, 156, 158, 160 # Начало роста
    ]
    print(f"Цены: {bullish_prices}")
    
    # Вызываем наши методы для расчета
    rsi_value_bull = analyzer.calculate_rsi(bullish_prices, period=14)
    macd_result_bull = analyzer.calculate_macd(bullish_prices)

    print(f"Результат RSI: {rsi_value_bull}")
    print(f"Результат MACD: {macd_result_bull}")