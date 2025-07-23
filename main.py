# --- Импорты ---
from input_collection.data_collector import DataCollector
from processing_analysis.technical_analyzer import TechnicalAnalyzer
from processing_analysis.strategy_engine import StrategyEngine
from processing_analysis.risk_manager import RiskManager
from decision_execution.trade_decision_engine import TradeDecisionEngine
from decision_execution.ibkr_trading import IbkrTraderSimulator
from monitoring_feedback.reporter import Reporter
import time

def run_single_stock_analysis(symbol: str, account_balance: float):
    """
    Выполняет полный цикл анализа для ОДНОГО тикера.
    """
    print(f"\n{'='*20} Анализ тикера: {symbol.upper()} {'='*20}")
    
    # --- Шаг 1: Сбор данных ---
    collector = DataCollector()
    price_history = collector.get_price_history(symbol, period="3mo") # Используем 3 месяца для большей точности MACD
    
    # Defensive check: Убедимся, что данные вообще пришли
    if price_history is None or price_history.empty:
        print(f"Не удалось получить данные для {symbol}. Пропускаем.")
        return 

    current_price = price_history.iloc[-1]
    price_list = price_history.tolist()
    print(f"Шаг 1: Успешно. Текущая цена: {current_price:.2f}")

    # --- Шаг 2: Технический анализ ---
    print("\nШаг 2: Технический анализ данных...")
    analyzer = TechnicalAnalyzer()

    # Рассчитываем ОБА индикатора
    rsi_value = analyzer.calculate_rsi(price_list, period=14)
    macd_result = analyzer.calculate_macd(price_list)

    # Упаковываем все результаты в единый словарь для передачи "мозгу"
    analysis_data = {
        'rsi': rsi_value,
        'macd': macd_result
    }
    print(f"Шаг 2: Результаты анализа: {analysis_data}")
    
    # --- Шаг 3: Стратегия ---
    print("\nШаг 3: Принятие решения на основе стратегии...")
    strategy_eng = StrategyEngine()
    signal = strategy_eng.run_all_strategies(analysis_data)
    # Сигнал уже будет содержать пояснение, если он не HOLD
    
    # --- Шаги 4-7 ---
    # Defensive check: Запускаем риск-менеджер и все последующие шаги ТОЛЬКО если есть активный сигнал
    if signal != "HOLD":
        print("\nШаг 4: Расчет рисков и размера позиции...")
        risk_man = RiskManager(account_balance=account_balance, risk_per_trade_pct=0.02)
        risk_parameters = risk_man.manage_trade(signal, current_price)

        print("\nШаг 5: Формирование торгового приказа...")
        decision_eng = TradeDecisionEngine()
        trade_order = decision_eng.make_decision(signal, symbol, current_price, risk_parameters)
        
        print("\nШаг 6: Исполнение сделки...")
        trader = IbkrTraderSimulator()
        trader.connect()
        trader.execute_trade(trade_order)
        trader.disconnect()
            
        print("\nШаг 7: Запись результата в журнал...")
        reporter = Reporter()
        reporter.log_trade(trade_order)
    else:
        # Если сигнала нет, мы просто сообщаем об этом и завершаем анализ для этого тикера
        print("\nШаги 4-7: Пропущены, так как нет активного торгового сигнала.")


# --- Точка входа в программу ---
def main():
    """
    Главная функция-оркестратор. Запускает анализ для списка тикеров.
    """
    print("--- ЗАПУСК ГЛАВНОГО СКАНЕРА РЫНКА (Стратегия: RSI + MACD Crossover) ---")
    
    ACCOUNT_BALANCE = 10000
    stocks_to_scan = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOG", "AMD", "SPY"]
    
    print(f"Будут проанализированы следующие акции: {stocks_to_scan}")

    # Главный цикл
    for stock_symbol in stocks_to_scan:
        try:
            run_single_stock_analysis(stock_symbol, ACCOUNT_BALANCE)
            time.sleep(2) # Пауза между запросами к API
        except Exception as e:
            # Defensive check: Ловим ЛЮБЫЕ непредвиденные ошибки для одного тикера,
            # чтобы весь сканер не остановился.
            print(f"!!! КРИТИЧЕСКАЯ НЕПРЕДВИДЕННАЯ ОШИБКА при анализе {stock_symbol}: {e} !!!")
            continue

    print(f"\n--- СКАНИРОВАНИЕ РЫНКА ЗАВЕРШЕНО ---")

if __name__ == '__main__':
    main()