# Код для strategy_engine.py (полный, правильный)
class StrategyEngine:
    def get_signal_from_score(self, final_score: int) -> str:
        print(f"[Strategy Engine]: Получена финальная оценка для принятия решения: {final_score}.")
        BUY_THRESHOLD, SELL_THRESHOLD = 7, -7
        if final_score >= BUY_THRESHOLD: print(f"[Strategy Engine]: Оценка >= {BUY_THRESHOLD}. Сгенерирован сигнал BUY."); return "BUY"
        elif final_score <= SELL_THRESHOLD: print(f"[Strategy Engine]: Оценка <= {SELL_THRESHOLD}. Сгенерирован сигнал SELL."); return "SELL"
        else: print(f"[Strategy Engine]: Оценка в нейтральной зоне. Сигнала нет."); return "HOLD"