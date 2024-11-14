import csv
from dsmltf import count_words, spam_probability, f1_score
from nltk import pos_tag

def word_probabilities(counts: list[tuple], total_spams: int, total_non_spams: int, k: float = 0.5) -> list[tuple]:
    """Вычисление вероятностей слов"""
    smoothed = []
    for w in counts:
        # Сглаженные вероятности для слова, принадлежащего к спаму или хэму
        smoothed_spam_prob = (w[1] + k) / (total_spams + 2 * k)
        smoothed_ham_prob = (w[2] + k) / (total_non_spams + 2 * k)
        smoothed.append((w[0], smoothed_spam_prob, smoothed_ham_prob))
        
    return smoothed


def make_data() -> list:
    """Загрузка данных из CSV файла."""
    with open("spam_ham_dataset.csv", encoding="utf-8") as f:
        data = []
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            label = int(row['label_num'])
            data.append([text, label])  # Пара (текст, метка)
    return data





def test(words: list[tuple], dataset: list) -> tuple:
    """Тестирование модели на тестовом наборе данных и вычисление метрик."""
    true_pos, false_pos, false_neg, true_neg = 0, 0, 0, 0
    for i in dataset:
        predicted = round(spam_probability(words, i[0]))  # 0 или 1
        actual = i[1]  # Реальная метка (0 или 1)

        if predicted == 1 and actual == 1:
            true_pos += 1
        elif predicted == 1 and actual == 0:
            false_pos += 1
        elif predicted == 0 and actual == 1:
            false_neg += 1
        elif predicted == 0 and actual == 0:
            true_neg += 1

    # Вычисление F1-меры
    f1 = f1_score(true_pos, false_pos, false_neg)

   
    accuracy = (true_pos + true_neg) / len(dataset)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    return true_pos, false_pos, false_neg, true_neg, f1, accuracy, precision, recall

def main() -> None:
    dataset = make_data()

    # Количество спамных и не-спамных сообщений в общем наборе данных
    spam_count = len([i for i in dataset if i[1]])
    ham_count = len(dataset) - spam_count
    print(f"Количество спамов в общем наборе: {spam_count}")
    print(f"Количество не-спамов в общем наборе: {ham_count}")


    # Тестовая выборка 
    test_data = dataset[-2000:]

    test_spam_count = len([i for i in test_data if i[1] == 1])
    print(f"Количество спамных сообщений в тестовой выборке: {test_spam_count}")
   
    # Тренировочные данные
    train_set = count_words(dataset[:-2000])



    # Словарь с определением части речи слова
    tagged_keys = pos_tag(train_set.keys())

    # Фильтруем данные (без прилагательных)
    train_set_filtered = {}
    for key, tag in tagged_keys:
        if tag not in ('JJ', 'JJR', 'JJS'):  # Исключаем прилагательные
            if key not in train_set_filtered:
                train_set_filtered[key] = [0, 0]  # Инициализация если ключ не существует
            train_set_filtered[key] = train_set[key]
    
   

    # Самые часто встречающиеся слова
    words = sorted(train_set_filtered, key=lambda x: train_set_filtered[x][0] if len(x) >= 5 else 0)[-7:]
    words = [(i, train_set_filtered[i][0] / spam_count, train_set_filtered[i][1] / ham_count if train_set_filtered[i][1] / ham_count else 0.01) for i in words]

    # Для отслеживания наилучшего k
    best_k = None
    best_f1 = 0  

    # Тестирование для разных значений k
    print("\nРезультаты для различных значений k:")
    for k in [100, 50, 1, 0, 0.1, 0.01]:  
        print(f"\nТестирование с k={k}:")
        
        # 1. Без сглаживания
        true_pos, false_pos, false_neg, true_neg, f1_no_smoothing, accuracy_no_smoothing, precision_no_smoothing, recall_no_smoothing = test(words, test_data)
        print(f"F1-мера (без сглаживания): {f1_no_smoothing}")
        print(f"Точность (без сглаживания): {accuracy_no_smoothing}")
        print(f"Precision (Точность): {precision_no_smoothing}")
        print(f"Recall (Полнота): {recall_no_smoothing}")
        print(f"True Positives: {true_pos}")
        print(f"False Positives: {false_pos}")
        print(f"False Negatives: {false_neg}")
        print(f"True Negatives: {true_neg}")

        # 2. С сглаживанием
        smoothed_words = word_probabilities([(i[0], train_set_filtered[i[0]][0], train_set_filtered[i[0]][1]) for i in words], spam_count, ham_count, k)
        true_pos, false_pos, false_neg, true_neg, f1_with_smoothing, accuracy_with_smoothing, precision_with_smoothing, recall_with_smoothing = test(smoothed_words, test_data)
        print(f"F1-мера (с сглаживанием, k={k}): {f1_with_smoothing}")
        print(f"Точность (с сглаживанием, k={k}): {accuracy_with_smoothing}")
        print(f"Precision (Точность): {precision_with_smoothing}")
        print(f"Recall (Полнота): {recall_with_smoothing}")
        print(f"True Positives: {true_pos}")
        print(f"False Positives: {false_pos}")
        print(f"False Negatives: {false_neg}")
        print(f"True Negatives: {true_neg}")

        # Отслеживаем наилучший k по F1-мере
        if f1_with_smoothing > best_f1:
            best_f1 = f1_with_smoothing
            best_k = k

        #print("\nЧастые слова с вероятностями для k=100:")
       # for word in smoothed_words:
          #  print(f"Слово: {word[0]} | Spam: {word[1]:.4f} | Ham: {word[2]:.4f}")
        
        #print("\nЧастые слова без сглаживания")
        #for word in words:
            #print(f"Слово: {word[0]} | Spam: {word[1]:.4f} | Ham: {word[2]:.4f}")
        

    print(f"\nНаилучшее значение k: {best_k} с F1-мерой {best_f1}")

if __name__ == "__main__":
    main()
