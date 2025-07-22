import logging
from typing import Dict, Any

# Эта часть кода выполняется один раз при первом импорте файла.
# Она настраивает "глобальный" логгер для нашего приложения.
logging.basicConfig(
    level=logging.INFO, # Минимальный уровень сообщений для записи. INFO, WARNING, ERROR будут записаны.
    format='%(asctime)s - %(levelname)s - %(message)s', # Формат строки: ВРЕМЯ - УРОВЕНЬ - СООБЩЕНИЕ
    filename='trade_history.log', # Имя файла, куда будут сохраняться логи.
    filemode='a' # 'a' (append) - добавлять в конец файла. 'w' (write) - перезаписывать каждый раз.
)

# Мы создаем именованный логгер. Это хорошая практика, если в будущем у вас будет несколько логгеров.
trade_logger = logging.getLogger('TradeLogger')

class Reporter:
    """
    Отвечает за логирование всех важных событий, в особенности
    совершенных сделок, в текстовый файл для последующего анализа.
    """
    def log_trade(self, trade_order: Dict[str, Any]):
        """
        Записывает информацию о сделке в лог-файл.
        """
        if trade_order.get('decision') == 'HOLD':
            # Нам не нужно логировать бездействие
            return

        # Формируем красивую и информативную строку для записи
        log_message = (
            f"TRADE EXECUTED: "
            f"Action={trade_order.get('decision')}, "
            f"Symbol={trade_order.get('symbol')}, "
            f"Quantity={trade_order.get('quantity')}, "
            f"Price=~{trade_order.get('estimated_price')}, "
            f"StopLoss={trade_order.get('stop_loss')}, "
            f"TakeProfit={trade_order.get('take_profit')}"
        )
        
        # Используем наш логгер для записи сообщения с уровнем INFO
        trade_logger.info(log_message)
        print("[Reporter]: Информация о сделке записана в trade_history.log")