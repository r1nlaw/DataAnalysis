import random
import re
import nltk

nltk.download("punkt")

def generate_text_with_regex(text: str, n: int = 3, max_retries: int = 100) -> str:
    # Токенизация текста
    words = nltk.word_tokenize(text)

    # Фильтруем токены, чтобы убрать знаки препинания
    words = [word for word in words if word.isalpha()]  # Оставляем только слова

    # Словарь для хранения переходов между (n-1)-граммами
    transitions = {}

    # Генерация (n-1)-грамм
    for i in range(len(words) - n + 1):
        context = tuple(words[i:i + n - 1])  # (n-1)-грамма
        next_word = words[i + n - 1]  # Следующее слово

        # Добавляем переходы для контекста в словарь
        if context not in transitions:
            transitions[context] = []
        transitions[context].append(next_word)

    # Проверка на пустоту словаря transitions
    if not transitions:
        return "Ошибка: В тексте недостаточно данных для генерации текста."

    # Генерация текста
    # Сначала выбираем случайный контекст
    start_word = random.choice(list(transitions.keys()))
    
    result = list(start_word)
    
    while True:
        # Получаем контекст из последних n-1 слов
        context = tuple(result[-(n - 1):])

        # Строим регулярное выражение для поиска продолжений
        context_str = " ".join(context)
        pattern = r"\b" + re.escape(context_str) + r"\s+(\w+)"  # Ищем следующее слово после контекста

        # Находим все продолжения для текущего контекста в исходном тексте
        matches = re.findall(pattern, text)

        if matches:
            next_word = random.choice(matches)  # Выбираем случайное продолжение

            # Игнорируем знаки препинания, если они попадают в следующую выборку
            while next_word in ['.', '!', '?', ',', ';', ':']:
                # Ищем следующее слово, которое не является знаком препинания
                matches = re.findall(pattern, text)
                if matches:
                    next_word = random.choice(matches)
                else:
                    break
            result.append(next_word)
        else:
            break  # Если нет продолжений, завершаем генерацию

        # Проверка, если сгенерированное слово является знаком препинания — завершаем
        if next_word in ['.', '!', '?']:
            break

    # Объединяем результат в строку
    return ' '.join(result)

def main() -> None:
    with open("text.txt", "r+", encoding="UTF8") as f:
        text = f.read()
        print(f"Генерация текста (Trigram): {generate_text_with_regex(text, 3)}")
        print(f"Генерация текста (Fourgram): {generate_text_with_regex(text, 4)}")

      

if __name__ == "__main__":
    main()
