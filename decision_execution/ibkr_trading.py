from typing import Dict, Any

class IbkrTraderSimulator:
    """
    ЭТО СИМУЛЯТОР!
    Он имитирует работу реального модуля для отправки ордеров брокеру.
    Он имеет те же методы, но вместо реальных действий выводит информацию в консоль.
    Это БЕЗОПАСНЫЙ способ тестирования всей цепочки логики.
    """
    def __init__(self, connection_params: Dict = None):
        print("\n[IBKR Simulator]: Инициализация симулятора торгового модуля...")
        self.is_connected = False

    def connect(self):
        print("[IBKR Simulator]: Попытка 'подключения' к торговому терминалу...")
        self.is_connected = True
        print("[IBKR Simulator]: 'Подключение' успешно установлено.")

    def execute_trade(self, trade_order: Dict[str, Any]):
        """
        "Исполняет" торговый приказ.
        (Это обновленная версия метода)
        """
        if not self.is_connected:
            print("[IBKR Simulator]: ОШИБКА! Невозможно исполнить ордер. Нет подключения.")
            return

        if trade_order.get('decision') == 'HOLD':
            print("[IBKR Simulator]: Получено решение 'HOLD'. Никаких действий не требуется.")
            return
        
        # Если есть активный ордер, "исполняем" его
        print("-" * 30)
        print("[IBKR Simulator]: ПОЛУЧЕН НОВЫЙ ТОРГОВЫЙ ПРИКАЗ!")
        print(f"  -> ДЕЙСТВИЕ:    {trade_order.get('decision')}")
        print(f"  -> ТИКЕР:       {trade_order.get('symbol')}")
        print(f"  -> КОЛ-ВО:      {trade_order.get('quantity')}")
        print(f"  -> ТИП ОРДЕРА:  {trade_order.get('order_type')}")
        print(f"  -> ЦЕНА ВХОДА:  ~{trade_order.get('estimated_price')}")
        print(f"  -> STOP-LOSS:   {trade_order.get('stop_loss')}")
        print(f"  -> TAKE-PROFIT: {trade_order.get('take_profit')}")
        print("[IBKR Simulator]: Ордер отправлен на 'исполнение'...")
        print("-" * 30)

    def disconnect(self):
        print("[IBKR Simulator]: 'Отключение' от торгового терминала.")
        self.is_connected = False