import csv
from dsmltf import knn_classify, k_neighbours_classify

# спарсим данные с csv файла
def load_dataset(lat_min, lat_max, lon_min, lon_max) -> list:
    res = []
    with open("dataset_earthquake.csv", "r") as f:
        for row in csv.reader(f, delimiter=','):
            try:
                latitude = float(row[1])
                longitude = float(row[2])
                magnitude = float(row[4])
                
                # Фильтрация по диапазонам
                if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                    res.append([latitude, longitude, magnitude])
            except:
                continue
    return res
    
   
        
def main():

    lat_min, lat_max = 24.396308, 45.551483  # широта
    lon_min, lon_max = 122.93457, 153.986672  # долгота

    # Загрузка данных с учетом заданного региона
    data = load_dataset(lat_min, lat_max, lon_min, lon_max)
    # датасеты приведенные в нужный формат
    data0 = [(i[:-1], i[-1]) for i in data]
    data1 = [(i[:-1], round(i[-1])) for i in data]
    unique = {i[1] for i in data1}
    print(len(unique))
    
   
    
    # словарь с результатами
    dict0 = k_neighbours_classify(11, data0[:100])
    dict1 = k_neighbours_classify(11, data1[:100])
    print(dict0)
    print(f"Количество данных в data0: {len(data0)}")
    print(f"Количество данных в data1: {len(data1)}")

    # проверка деления на ноль
    a0 = [dict0[i][0] / dict0[i][1] if dict0[i][1] != 0 else 0 for i in range(1, 12)]
    a1 = [dict1[i][0] / dict1[i][1] if dict1[i][1] != 0 else 0 for i in range(1, 12)]

    # лучшие k
    k0 = a0.index(max(a0)) + 1
    k1 = a1.index(max(a1)) + 1

    latitude, longitude = None, None
    while latitude is None or longitude is None:
        try:
            values = input("Введите широту и долготу: ").split()
            if len(values) != 2:
                print("Введите два числа: широту и долготу.")
                continue
            latitude, longitude = map(float, values)
        except ValueError:
            print("Введите действительные числовые значения.")

    # проверяем есть ли соседи
    if len(data0) == 0 or len(data1) == 0:
        print("Недостаточно данных для классификации.")
        return

    # Классификация
    try:
        result0 = knn_classify(k0, data0, (latitude, longitude))
        result1 = knn_classify(k1, data1, (latitude, longitude))
        print("Результат классификации (данные 0):", result0)
        print("Результат классификации (данные 1):", result1)
    except IndexError:
        print("Нет соседей для указанных координат.")

if __name__ == "__main__":
    main()
