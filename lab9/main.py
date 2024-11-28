import csv
from dsmltf import scale, KMeans, bottom_up_cluster, generate_clusters, get_values
import numpy as np
import matplotlib.pyplot as plt  

def make_data() -> list:
    """Парсим данные из CSV и создаем словари для категориальных данных"""
    
    with open("flowers.csv", "r", encoding="UTF-8") as f:
        data = []
        
        # Словари для преобразования категориальных данных
        color_type = {
            "Red": 0,
            "Blue": 1,
            "Yellow": 2,
            "Green": 3,
            "White": 4,
            "Orange": 5,
            "Pink": 6,
            "Lavender": 7,
            "Purple": 8,
            "Sky Blue": 9
        }
        
        flower_type = {
            "Tropical": 0,
            "Meadow": 1,
            "Garden": 2,
            "Mountain": 3
            
        }
        
        
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "Color":
                continue
            
            
            color = color_type.get(row[0], -1)  # Преобразуем цвет в число
            flower_class = flower_type.get(row[1], -1)  # Преобразуем тип цветка в число
           
            # Преобразуем размеры лепестков в числа
            try:
                petal_size = float(row[2])  # Размер лепестка
                petal_width = float(row[3])  # Ширина лепестка
            except ValueError:
                print(f"Некорректные данные в строке: {row}")
                continue  # Пропускаем строки с некорректными значениями
            
            # Добавляем данные в список
            data.append([color, flower_class, petal_size, petal_width])
    
    return data

def calculate_sse(data, k):
    """Функция для вычисления суммы квадратов ошибок для заданного k"""
    # Кластеризация методом K-средних
    clast = KMeans(k)  
    clast.train(data)
    
    # Для каждой точки вычисляем расстояние до ее кластера и суммируем квадраты этих расстояний
    sse = 0
    for point in data:
        cluster_id = clast.classify(point)
        centroid = clast.means[cluster_id]
        sse += np.linalg.norm(np.array(point) - np.array(centroid)) ** 2
    return sse

def main() -> None:
    
    data_set = make_data()
    print(f"Количество обработанных строк: {len(data_set)}")

    scale_data = scale(data_set[:200])  # Шкалируем данные 
    
    # Вычисляем SSE для различных значений k
    sse_values = []
    max_k = 11  # Максимальное количество кластеров
    for k in range(1, max_k + 1):
        sse = calculate_sse(scale_data, k)
        sse_values.append(sse)
    

    
    plt.plot(range(1, max_k + 1), sse_values, marker='o')
    plt.title("Поиска оптимального количества кластеров")
    plt.xlabel("Количество кластеров (k)")
    plt.ylabel("Сумма квадратов ошибок (SSE)")
    plt.show()

   
    optimal_k = np.argmin(np.diff(sse_values)) + 2  # Определение k, где происходит резкое изменение
    diff_sse = np.diff(sse_values)
    print(diff_sse)
    print(f"Оптимальное количество кластеров: {optimal_k}")

    # Кластеризация методом K-средних с оптимальным k
    clast = KMeans(optimal_k)  
    clast.train(scale_data)

    # Получаем центроиды кластеров
    centroids = clast.means 
    print("Центроиды кластеров в k средних:", centroids)

    # Подсчитываем количество элементов в каждом кластере
    clusters_kmeans = {i: 0 for i in range(optimal_k)}  # Создаем словарь для подсчета объектов в каждом кластере

    # Для каждой точки находим ближайший центроид и увеличиваем соответствующий счетчик
    for point in scale_data:
        # Вычисляем расстояния до всех центроидов
        distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
        # Находим индекс ближайшего центроида
        cluster_index = np.argmin(distances)
        clusters_kmeans[cluster_index] += 1

    print("\nКоличество объектов в каждом кластере K-средних:")
    for i, count in clusters_kmeans.items():
        print(f"Кластер {i}: {count}")

    # Восходящая кластеризация
    base_claster = bottom_up_cluster(scale_data)
    print("\nРезультаты восходящей кластеризации:")
    
    # Генерируем и выводим кластеры с нумерацией
    for i, cluster in enumerate(generate_clusters(base_claster, 5), 0):  
        values_in_cluster = get_values(cluster)
        print(f"Кластер {i}: {len(values_in_cluster)}")

if __name__ == "__main__":
    main()
