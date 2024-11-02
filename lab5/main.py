import random
import math

# Генерация датасета
data = []
for _ in range(30):
    month = random.randint(1, 12)  # Месяц от 1 до 12
    weather = random.randint(0, 5)  # Погода от 0 до 5
    fish_type = random.randint(1, 5)  # Вид рыбы от 1 до 5
    gear = random.randint(1, 3)  # Снасть от 1 до 3
    # Статус удачи рыбалки генерируем случайно
    status = random.randint(0, 1)  # 0 - неудачно, 1 - удачно
    data.append([month, weather, fish_type, gear, status])

# Выводим сгенерированный датасет
for row in data:
    print(row)


# Нормализация данных по Min-Max методу
def min_max_scale(data):
    scaled_data = []
    for col in range(len(data[0]) - 1):  # Не нормализуем столбец со статусом
        col_values = [row[col] for row in data]
        min_val, max_val = min(col_values), max(col_values)
        scaled_column = [(x - min_val) / (max_val - min_val) for x in col_values]
        for i in range(len(data)):
            data[i][col] = scaled_column[i]
    return data


scaled_data = min_max_scale(data)

# Выводим нормализованный датасет
for row in scaled_data:
    print(row)

def split_data(data, train_size=0.9):
    train_data = data[:int(len(data) * train_size)]
    test_data = data[int(len(data) * train_size):]
    return train_data, test_data

train_data, test_data = split_data(scaled_data)

# Проверяем, что разделение выполнено
print(f"Обучающая выборка: {len(train_data)} записей")
print(f"Тестовая выборка: {len(test_data)} записей")

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(row, weights):
    z = weights[0]
    for i in range(len(row) - 1):
        z += weights[i + 1] * row[i]
    return sigmoid(z)

def train_logistic_regression(train_data, learning_rate, n_epochs):
    weights = [0.0] * len(train_data[0])
    for epoch in range(n_epochs):
        for row in train_data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += learning_rate * error * row[i]
    return weights

# Обучаем модель
learning_rate = 0.01
n_epochs = 500
weights = train_logistic_regression(train_data, learning_rate, n_epochs)

print(f"Обученные веса: {weights}")

def classify(prediction):
    return 1 if prediction >= 0.5 else 0

# Оценка точности на тестовой выборке
def evaluate_model(test_data, weights):
    correct_predictions = 0
    for row in test_data:
        prediction = predict(row, weights)
        predicted_label = classify(prediction)
        if predicted_label == row[-1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_data)
    return accuracy

accuracy = evaluate_model(test_data, weights)
print(f"Точность модели на тестовой выборке: {accuracy * 100:.2f}%")
