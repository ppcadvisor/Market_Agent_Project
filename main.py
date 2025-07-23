# Код для main.py (полный, правильный)
from input_collection.data_collector import DataCollector
from input_collection.fundamental_parser import FundamentalParser
from processing_analysis.technical_analyzer import TechnicalAnalyzer
from processing_analysis.scoring_engine import ScoringEngine
from processing_analysis.strategy_engine import StrategyEngine
from processing_analysis.risk_manager import RiskManager
from decision_execution.trade_decision_engine import TradeDecisionEngine
from decision_execution.ibkr_trading import IbkrTraderSimulator
from monitoring_feedback.reporter import Reporter
import time
def run_single_stock_analysis(symbol: str, account_balance: float):
    print(f"\n{'='*20} Анализ тикера: {symbol.upper()} {'='*20}")
    price_collector, price_history = DataCollector(), None
    price_history = price_collector.get_price_history(symbol, period="3mo")
    if price_history is None: return
    fundamental_parser = FundamentalParser()
    fundamental_data = fundamental_parser.get_fundamental_data(symbol)
    print(f"Шаг 1: Сбор данных завершен. Текущая цена: {price_history.iloc[-1]:.2f}")
    analyzer = TechnicalAnalyzer()
    rsi_value = analyzer.calculate_rsi(price_history.tolist())
    macd_result = analyzer.calculate_macd(price_history.tolist())
    print("\nШаг 3: Агрегация и оценка сигналов...")
    full_analysis_data = {'rsi': rsi_value, 'macd': macd_result, 'fundamental': fundamental_data}
    strategy_weights = {'technical': 0.7, 'fundamental': 0.3}
    scorer = ScoringEngine(weights=strategy_weights)
    final_score = scorer.calculate_final_score(full_analysis_data)
    print("\nШаг 4: Принятие решения на основе оценки...")
    strategy_eng = StrategyEngine()
    signal = strategy_eng.get_signal_from_score(final_score)
    if signal != "HOLD":
        print("\nШаг 5: Расчет рисков и размера позиции...")
        risk_man = RiskManager(account_balance=account_balance, risk_per_trade_pct=0.02)
        risk_parameters = risk_man.manage_trade(signal, price_history.iloc[-1])
        print("\nШаг 6: Формирование торгового приказа...")
        decision_eng = TradeDecisionEngine()
        trade_order = decision_eng.make_decision(signal, symbol, price_history.iloc[-1], risk_parameters)
        print("\nШаг 7: Исполнение сделки...")
        trader = IbkrTraderSimulator(); trader.connect(); trader.execute_trade(trade_order); trader.disconnect()
        print("\nШаг 8: Запись результата в журнал...")
        reporter = Reporter(); reporter.log_trade(trade_order)
    else: print("\nШаги 5-8: Пропущены, так как нет активного торгового сигнала.")
def main():
    print("--- ЗАПУСК ГЛАВНОГО СКАНЕРА РЫНКА (Стратегия: Взвешенная) ---")
    ACCOUNT_BALANCE = 10000; stocks_to_scan = ["AAPL", "MSFT", "NVDA", "TSLA"]
    for stock_symbol in stocks_to_scan:
        try: run_single_stock_analysis(stock_symbol, ACCOUNT_BALANCE); time.sleep(2)
        except Exception as e: print(f"!!! КРИТИЧЕСКАЯ НЕПРЕДВИДЕННАЯ ОШИБКА при анализе {stock_symbol}: {e} !!!"); continue
    print(f"\n--- СКАНИРОВАНИЕ РЫНКА ЗАВЕРШЕНО ---")
if __name__ == '__main__': main()