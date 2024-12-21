import random
import numpy as np



# Функция сигмоида, которая будет использоваться для активации нейронов
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Функция для кодирования метки класса в one-hot представление (например, 3 -> [0, 0, 1, 0, 0])
def one_hot_encode(label, num_classes=10):
    return np.eye(num_classes)[label]

# Функция для прямого прохода через нейронную сеть
def forward_pass(data, weights, biases):
    activations = [data]  # Список для хранения активаций (входных и промежуточных)
    for w, b in zip(weights, biases):
        z = np.dot(activations[-1], w.T) + b  # Вычисляем линейную комбинацию входных данных
        a = sigmoid(z)  # Применяем функцию активации (сигмоиду)
        activations.append(a)  # Сохраняем активацию для следующего слоя
    return activations  # Возвращаем список всех активаций

# Функция для шага обучения 
def train_iteration(data, weights, biases, learning_rate=0.1, noise_level=0.1, num_classes=10):
    # Добавляем шум к изображениям на входе (псевдослучайные изменения пикселей)
    X_noisy = [add_noise(d[0], noise_level) for d in data]
    X = np.array(X_noisy, dtype=np.float32)  # Преобразуем входные данные в массив NumPy
    
    y = np.array([d[1] for d in data], dtype=np.int32)  # Извлекаем метки
    y_onehot = np.array([one_hot_encode(label, num_classes) for label in y], dtype=np.float32)  # Преобразуем метки в one-hot формат

    activations = forward_pass(X, weights, biases)  # Выполняем прямой проход
    output = activations[-1]  # Получаем выход сети (последний слой)

    # Начинаем обратное распространение ошибки
    errors = [None] * len(weights)  # Список для хранения ошибок на каждом слое
    errors[-1] = (y_onehot - output) * output * (1 - output)  # Ошибка на выходном слое (по формуле градиента для сигмоиды)

    for i in range(len(weights) - 2, -1, -1):  # Пройдем по всем слоям в обратном порядке
        # Вычисляем ошибку для каждого слоя
        errors[i] = errors[i + 1].dot(weights[i + 1]) * activations[i + 1] * (1 - activations[i + 1])

    # Вычисляем градиенты для обновления весов и смещений
    gradients = [None] * len(weights)
    gradients_bias = [None] * len(weights)

    for i in range(len(weights)):
        gradients[i] = np.dot(errors[i].T, activations[i])  # Градиент для весов
        gradients_bias[i] = np.sum(errors[i], axis=0)  # Градиент для смещений

    # Обновляем веса и смещения с использованием градиентного спуска
    for i in range(len(weights)):
        weights[i] += learning_rate * gradients[i]
        biases[i] += learning_rate * gradients_bias[i]

    return weights, biases  # Возвращаем обновленные веса и смещения

# Функция для инициализации сети с случайными весами и нулями для смещений
def initialize_network(input_size, layers):
    weights, biases = [], []
    for i, size in enumerate(layers):
        prev_size = input_size if i == 0 else layers[i - 1]
        weights.append(np.random.rand(size, prev_size) - 0.5)  # Случайные веса с центровкой вокруг 0
        biases.append(np.zeros(size))  # Смещения инициализируем нулями
    return weights, biases  # Возвращаем списки весов и смещений

# Функция для добавления шума к изображению (меняем случайные пиксели)
def add_noise(pattern, noise_level=0.1):
    noisy_pattern = pattern.copy()
    num_elements = len(noisy_pattern)
    num_to_change = int(num_elements * noise_level)  # Количество пикселей, которые нужно изменить
    indices_to_change = random.sample(range(num_elements), num_to_change)  # Случайно выбираем индексы
    for index in indices_to_change:
        noisy_pattern[index] = 1 - noisy_pattern[index]  # Меняем пиксель на противоположный
    return noisy_pattern  # Возвращаем зашумленное изображение

# Функция для предсказания меток для тестовых данных
def predict(test_data, weights, biases):
    X = np.array([td[0] for td in test_data], dtype=np.float32)  # Извлекаем тестовые данные
    activations = forward_pass(X, weights, biases)  # Выполняем прямой проход
    outputs = activations[-1]  # Получаем выходной слой
    predictions = np.argmax(outputs, axis=1)  # Выбираем класс с максимальной вероятностью
    actual = np.array([td[1] for td in test_data])  # Извлекаем реальные метки
    errors = np.sum(predictions != actual)  # Считаем количество ошибок
    return errors, predictions, actual  # Возвращаем количество ошибок, предсказания и реальные метки

# Функция для расширения изображения до нового размера
def expand_to_larger_size(image, new_size=(10, 10)):
    original_height, original_width = len(image), len(image[0])
    new_height, new_width = new_size
    scale_h, scale_w = new_height // original_height, new_width // original_width  # Масштабирование по осям

    expanded_image = [[0] * new_width for _ in range(new_height)]  # Создаем пустое изображение нового размера
    for i in range(original_height):
        for j in range(original_width):
            for m in range(i * scale_h, (i + 1) * scale_h):  # Размножаем значения пикселей по новому размеру
                for n in range(j * scale_w, (j + 1) * scale_w):
                    expanded_image[m][n] = image[i][j]
    return expanded_image  # Возвращаем расширенное изображение


# Базовые данные для цифр 5x5
digits_5x5 = [
    [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]],  # 0
    [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],  # 1
    [[0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0]],  # 2
    [[0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]],  # 3
    [[0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 1], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],  # 4
    [[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]],  # 5
    [[0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]],  # 6
    [[1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 1, 0]],  # 7
    [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]],  # 8
    [[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 1, 0], [0, 1, 1, 1, 0]]   # 9
]

# Создаем тестовую выборку для изображений размером 10x10
digits_10x10 = [expand_to_larger_size(digit, new_size=(10, 10)) for digit in digits_5x5]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X_5x5 = [np.array(digit, dtype=np.float32).flatten() for digit in digits_5x5]
X_10x10 = [np.array(digit, dtype=np.float32).flatten() for digit in digits_10x10]

# Инициализация сети для 5x5 данных
weights_5x5, biases_5x5 = initialize_network(25, [25, 10])
labeled_data_5x5 = list(zip(X_5x5, labels))

# Инициализация сети для 10x10 данных
weights_10x10, biases_10x10 = initialize_network(100, [100, 50, 10])
labeled_data_10x10 = list(zip(X_10x10, labels))

# Тестирование на зашумленных данных
test_data_noisy_5x5 = [add_noise(pattern, noise_level=0.1) for pattern in X_5x5]
test_set_5x5 = list(zip(test_data_noisy_5x5, labels))

test_data_noisy_10x10 = [add_noise(pattern, noise_level=0.1) for pattern in X_10x10]
test_set_10x10 = list(zip(test_data_noisy_10x10, labels))


# Обучение
for epoch in range(600):
    # Обучение на данных 5x5
    weights_5x5, biases_5x5 = train_iteration(labeled_data_5x5, weights_5x5, biases_5x5, learning_rate=0.1, noise_level=0.0)
    
    # Обучение на данных 10x10
    weights_10x10, biases_10x10 = train_iteration(labeled_data_10x10, weights_10x10, biases_10x10, learning_rate=0.1, noise_level=0.0)
    
    # Вывод на тестовой выборке после каждой эпохи
    if epoch % 50 == 0:
        # Для 5x5
        train_error_rate_5x5, train_predictions_5x5, train_actual_5x5 = predict(labeled_data_5x5, weights_5x5, biases_5x5)
        test_error_rate_5x5, test_predictions_5x5, test_actual_5x5 = predict(test_set_5x5, weights_5x5, biases_5x5)

        print(f"Epoch: {epoch} (5x5 Data)")
        print(f"Train Error Rate: {train_error_rate_5x5 / len(labeled_data_5x5)}, {train_error_rate_5x5} / {len(labeled_data_5x5)}")
        print(f"Test Error Rate: {test_error_rate_5x5 / len(test_set_5x5)}, {test_error_rate_5x5} / {len(test_set_5x5)}")
        print(f"Train Predictions: {train_predictions_5x5}")
        print(f"Train Actual: {train_actual_5x5}")
        print(f"Test Predictions: {test_predictions_5x5}")
        print(f"Test Actual: {test_actual_5x5}")
        print("")

        # Для 10x10
        train_error_rate_10x10, train_predictions_10x10, train_actual_10x10 = predict(labeled_data_10x10, weights_10x10, biases_10x10)
        test_error_rate_10x10, test_predictions_10x10, test_actual_10x10 = predict(test_set_10x10, weights_10x10, biases_10x10)

        print(f"Epoch: {epoch} (10x10 Data)")
        print(f"Train Error Rate: {train_error_rate_10x10 / len(labeled_data_10x10)}, {train_error_rate_10x10} / {len(labeled_data_10x10)}")
        print(f"Test Error Rate: {test_error_rate_10x10 / len(test_set_10x10)}, {test_error_rate_10x10} / {len(test_set_10x10)}")
        print(f"Train Predictions: {train_predictions_10x10}")
        print(f"Train Actual: {train_actual_10x10}")
        print(f"Test Predictions: {test_predictions_10x10}")
        print(f"Test Actual: {test_actual_10x10}")
        print("")
