# --- Импорты ---
from input_collection.data_collector import DataCollector # <-- Наш обновленный сборщик
from processing_analysis.technical_analyzer import TechnicalAnalyzer
from processing_analysis.strategy_engine import StrategyEngine
from processing_analysis.risk_manager import RiskManager
from decision_execution.trade_decision_engine import TradeDecisionEngine
from decision_execution.ibkr_trading import IbkrTraderSimulator
from monitoring_feedback.reporter import Reporter

def run_agent_cycle():
    """
    Имитирует один полный цикл работы агента, используя РЕАЛЬНЫЕ рыночные данные.
    """
    print("--- Запуск нового цикла работы агента ---")
    
    # --- ШАГ 1: СБОР РЕАЛЬНЫХ ДАННЫХ ---
    SYMBOL = "MSFT"  # Давайте анализировать Microsoft! Можете поменять на "AAPL", "GOOG", "NVDA" и т.д.
    ACCOUNT_BALANCE = 10000
    
    collector = DataCollector()
    price_history = collector.get_price_history(SYMBOL, period="2mo") # Загрузим данные за 2 месяца

    # Проверка, что данные получены успешно
    if price_history is None or price_history.empty:
        print("Не удалось получить данные. Завершение цикла.")
        return

    # Нам нужна цена и список цен для анализа
    current_price = price_history.iloc[-1] # Последняя цена в истории
    price_list = price_history.tolist()    # Преобразуем в обычный список для анализатора
    
    print(f"Шаг 1: Успешно. Тикер: {SYMBOL}. Текущая цена: {current_price:.2f}. Баланс: ${ACCOUNT_BALANCE}")

    # --- Остальные шаги остаются почти без изменений ---
    print("\nШаг 2: Технический анализ данных...")
    analyzer = TechnicalAnalyzer()
    rsi_value = analyzer.calculate_rsi(price_list, period=14)
    analysis_data = {'rsi': rsi_value}
    print(f"Результаты анализа: {analysis_data}")

    # ... и так далее
    print("\nШаг 3: Принятие решения на основе стратегии...")
    strategy_eng = StrategyEngine()
    signal = strategy_eng.run_all_strategies(analysis_data)
    print(f"Движок стратегий сгенерировал сигнал: {signal}")

    print("\nШаг 4: Расчет рисков и размера позиции...")
    if signal != "HOLD":
        risk_man = RiskManager(account_balance=ACCOUNT_BALANCE, risk_per_trade_pct=0.02)
        risk_parameters = risk_man.manage_trade(signal, current_price)
    else:
        risk_parameters = {}

    print("\nШаг 5: Формирование торгового приказа...")
    decision_eng = TradeDecisionEngine()
    trade_order = decision_eng.make_decision(signal, SYMBOL, current_price, risk_parameters)
    
    print("\nШаг 6: Исполнение сделки...")
    trader = IbkrTraderSimulator()
    trader.connect()
    trader.execute_trade(trade_order)
    trader.disconnect()
        
    print("\nШаг 7: Запись результата в журнал...")
    reporter = Reporter()
    reporter.log_trade(trade_order)

    print("\n--- Цикл работы агента завершен ---")

if __name__ == '__main__':
    run_agent_cycle()