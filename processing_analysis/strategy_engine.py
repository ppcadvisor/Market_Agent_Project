from typing import Dict, List, Optional

class StrategyEngine:
    """
    Применяет заранее определенные торговые стратегии к обработанным данным
    для генерации финальных торговых сигналов (BUY, SELL, HOLD).
    Это мозг, который решает, что делать на основе анализа.
    """

    def run_rsi_strategy(self, analysis_results: Dict) -> Optional[str]:
        """
        Простая стратегия, основанная только на RSI.

        :param analysis_results: Словарь с результатами анализа,
                                 например: {'rsi': 87.35, 'macd': -0.5}
        :return: Строку с сигналом "BUY", "SELL" или None, если сигнала нет.
        """
        # Извлекаем значение RSI из словаря.
        # .get() - безопасный способ получить значение. Если ключа 'rsi' нет, он вернет None.
        rsi_value = analysis_results.get('rsi')

        # Если по какой-то причине RSI не был рассчитан, мы не можем дать сигнал.
        if rsi_value is None:
            return None

        # --- Вот здесь и находится сама логика СТРАТЕГИИ ---
        if rsi_value > 70:
            return "SELL"  # Сигнал к продаже (ожидаем падения цены)
        
        if rsi_value < 30:
            return "BUY"   # Сигнал к покупке (ожидаем роста цены)
        
        # Если ни одно из условий не выполнилось, четкого сигнала нет.
        return None


    def run_all_strategies(self, analysis_results: Dict) -> str:
        """
        Запускает все доступные стратегии и выбирает первый сгенерированный сигнал.
        В будущем этот метод может стать сложнее (например, объединять сигналы).

        :param analysis_results: Словарь с результатами анализа.
        :return: Финальный сигнал: "BUY", "SELL" или "HOLD".
        """
        print("Запуск движка стратегий...")

        # В будущем здесь может быть цикл по списку разных стратегий
        # strategies = [self.run_rsi_strategy, self.run_macd_strategy, ...]
        
        signal = self.run_rsi_strategy(analysis_results)

        if signal:
            print(f"Стратегия RSI сгенерировала сигнал: {signal}")
            return signal
        
        # Если ни одна стратегия не дала сигнала, по умолчанию мы ничего не делаем.
        print("Ни одна из стратегий не сгенерировала активного сигнала.")
        return "HOLD"