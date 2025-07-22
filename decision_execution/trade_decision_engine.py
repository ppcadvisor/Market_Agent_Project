from typing import Dict, Any

class TradeDecisionEngine:
    """
    Принимает финальный сигнал и параметры от RiskManager,
    чтобы сформировать конкретный объект "приказа на сделку".
    """
    def make_decision(self, signal: str, symbol: str, current_price: float, risk_params: Dict) -> Dict[str, Any]:
        """
        Формирует словарь, описывающий сделку.

        :param signal: Сигнал "BUY", "SELL" или "HOLD".
        :param symbol: Тикер актива, например, "AAPL".
        :param current_price: Текущая цена актива.
        :param risk_params: Словарь с параметрами от RiskManager ('quantity', 'stop_loss'...).
        :return: Словарь с деталями ордера или решение "HOLD".
        """
        if signal == "HOLD" or not risk_params:
            return {'decision': 'HOLD'}

        # Формируем приказ, используя РАССЧИТАННЫЕ данные, а не заглушки
        trade_order = {
            'decision': signal,
            'symbol': symbol,
            'quantity': risk_params.get('quantity'),       # <-- Данные из RiskManager
            'order_type': 'MARKET',
            'stop_loss': risk_params.get('stop_loss'),    # <-- Данные из RiskManager
            'take_profit': risk_params.get('take_profit'),# <-- Данные из RiskManager
            'estimated_price': current_price
        }
        return trade_order