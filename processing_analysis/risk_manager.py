from typing import Dict, Any

class RiskManager:
    """
    Оценивает риски для сгенерированных сигналов и рассчитывает
    ключевые параметры для ордера: размер позиции, Stop-Loss и Take-Profit.
    """
    def __init__(self, account_balance: float, risk_per_trade_pct: float = 0.02):
        """
        :param account_balance: Общий баланс на торговом счете.
        :param risk_per_trade_pct: Процент от баланса, которым мы готовы рискнуть в одной сделке.
                                    0.02 означает 2% - это стандарт в индустрии.
        """
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.risk_per_trade_usd = self.account_balance * self.risk_per_trade_pct
        print(f"[Risk Manager]: Инициализирован. Макс. риск на сделку: ${self.risk_per_trade_usd:.2f}")

    def manage_trade(self, signal: str, current_price: float) -> Dict[str, Any]:
        """
        Рассчитывает параметры для безопасной сделки.

        :param signal: Сигнал "BUY" или "SELL".
        :param current_price: Текущая цена актива.
        :return: Словарь с параметрами: stop_loss, take_profit, quantity.
        """
        if signal not in ["BUY", "SELL"]:
            return {} # Возвращаем пустой словарь, если нет активного сигнала

        # --- Логика для сделки на покупку (BUY) ---
        if signal == "BUY":
            # 1. Устанавливаем Stop-Loss. Например, на 5% ниже текущей цены.
            # В реальных системах это может быть более сложный расчет (например, ниже последнего минимума).
            stop_loss_price = current_price * 0.95
            
            # 2. Устанавливаем Take-Profit. Например, с соотношением риск/прибыль 1:2.
            price_diff = current_price - stop_loss_price
            take_profit_price = current_price + (price_diff * 2)

            # 3. Рассчитываем размер позиции. ЭТО САМОЕ ВАЖНОЕ!
            # Размер позиции = (Сумма риска в $) / (Расстояние от цены входа до стоп-лосса в $)
            if price_diff == 0: return {} # Защита от деления на ноль
            position_size = self.risk_per_trade_usd / price_diff
            
            print(f"[Risk Manager]: Для BUY-сделки: Stop-Loss={stop_loss_price:.2f}, Take-Profit={take_profit_price:.2f}, Кол-во={int(position_size)}")
            return {
                'stop_loss': round(stop_loss_price, 2),
                'take_profit': round(take_profit_price, 2),
                'quantity': int(position_size) # Берем целое число акций
            }

        # --- Логика для сделки на продажу (SELL) ---
        if signal == "SELL":
            # 1. Устанавливаем Stop-Loss. Например, на 5% выше текущей цены.
            stop_loss_price = current_price * 1.05
            
            # 2. Устанавливаем Take-Profit.
            price_diff = stop_loss_price - current_price
            take_profit_price = current_price - (price_diff * 2)

            # 3. Рассчитываем размер позиции.
            if price_diff == 0: return {}
            position_size = self.risk_per_trade_usd / price_diff
            
            print(f"[Risk Manager]: Для SELL-сделки: Stop-Loss={stop_loss_price:.2f}, Take-Profit={take_profit_price:.2f}, Кол-во={int(position_size)}")
            return {
                'stop_loss': round(stop_loss_price, 2),
                'take_profit': round(take_profit_price, 2),
                'quantity': int(position_size)
            }
        
        return {}