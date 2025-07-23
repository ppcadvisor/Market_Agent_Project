import subprocess
import pandas as pd
from io import StringIO
from typing import List, Optional
import sys  # <-- ВАЖНО: Добавляем импорт sys

class IdeaGenerator:
    """
    Использует внешний инструмент (etorotrade) как скринер для генерации
    списка потенциально интересных акций для дальнейшего глубокого анализа.
    """
    def get_buy_recommendations(self) -> Optional[List[str]]:
        """
        Запускает etorotrade для получения списка акций с рекомендацией 'BUY'
        и возвращает список их тикеров.
        """
        try:
            # --- ИСПРАВЛЕНИЕ ---
            # Получаем ПОЛНЫЙ путь к python.exe из ТЕКУЩЕГО venv, чтобы избежать путаницы
            python_executable = sys.executable 
            # Теперь команда использует этот точный путь
            command = [python_executable, '-m', 'etorotrade.trade']
            # --------------------
            
            # Ввод пользователя: 't' (Trade Analysis), Enter, 'b' (BUY), Enter
            user_input = "t\nb\n"
            
            print("[Idea Generator]: Запускаю etorotrade... Это может занять до минуты.")
            
            # Запускаем процесс с ТАЙМАУТОМ.
            result = subprocess.run(
                command,
                input=user_input,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120
            )

            # Defensive Check 1: Проверяем, не завершился ли процесс с ошибкой
            if result.returncode != 0:
                print("[Idea Generator]: ОШИБКА! etorotrade завершился с ненулевым кодом.")
                print("--- Вывод ошибок (stderr) ---")
                print(result.stderr)
                print("-----------------------------")
                return None

            output = result.stdout
            print("[Idea Generator]: etorotrade успешно отработал. Начинаю парсинг вывода.")
            
            # Defensive Check 2: Ищем начало таблицы
            table_start_index = output.find('# TICKER')
            if table_start_index == -1:
                print("[Idea Generator]: ОШИБКА! Не удалось найти таблицу с результатами в выводе etorotrade.")
                print("--- Полный вывод (stdout) ---")
                print(output)
                print("-----------------------------")
                return []
                
            # Извлекаем текст таблицы
            table_text = output[table_start_index:]
            
            # Используем pandas для парсинга
            try:
                df = pd.read_csv(StringIO(table_text), sep='\s*\|\s*', engine='python', skipinitialspace=True)
                df = df.iloc[:, 1:-1]
                
                if 'TICKER' not in df.columns:
                    print("[Idea Generator]: ОШИБКА! В таблице отсутствует колонка 'TICKER'.")
                    return []

                tickers = df['TICKER'].dropna().astype(str).tolist()
                print(f"[Idea Generator]: Успешно. Получены {len(tickers)} кандидатов на покупку: {tickers}")
                return tickers
            except Exception as e:
                print(f"[Idea Generator]: ОШИБКА при парсинге таблицы pandas: {e}")
                return []

        except subprocess.TimeoutExpired:
            print("[Idea Generator]: ОШИБКА! Время ожидания ответа от etorotrade истекло (120 секунд).")
            return None
        except FileNotFoundError:
            # Эта ошибка теперь маловероятна, но оставляем ее для надежности
            print(f"[Idea Generator]: ОШИБКА! Не найден исполняемый файл Python: {sys.executable}")
            return None
        except Exception as e:
            print(f"[Idea Generator]: Произошла НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
            return None

# --- Тестовый блок для проверки ---
if __name__ == '__main__':
    generator = IdeaGenerator()
    recommendations = generator.get_buy_recommendations()
    
    if recommendations is not None:
        print("\n--- Финальный список рекомендованных акций ---")
        if recommendations:
            print(recommendations)
        else:
            print("Список пуст (сигналов не найдено).")
    else:
        print("\n--- Не удалось получить список акций из-за ошибки. ---")