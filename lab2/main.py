from math import erf, sqrt, pi, exp
import scipy.integrate as integrate
from scipy.integrate import quad

# Функция для вычисления значения функции распределения нормального распределения в точке x

def f_norm(count_tiles, mu=0, s=1):
    return (1 + erf((count_tiles - mu) / sqrt(2) / s)) / 2

# Функция для вычисления плотности вероятности нормального распределения в точке x
# mu - мат ожидание, s - сигма
def rho_norm(count_tiles, mu=0, s=1):
    return 1 / sqrt(2 * pi * s) * exp(-(count_tiles - mu)**2 / 2 / s**2)

# Функция для вычисления обратной функции распределения нормального распределения
# p - вероятность, mu0 - среднее значение, s - стандартное отклонение, t - точность
def inv_f_norm(p, mu0, s, t=0.001):
    if mu0 != 0 or s != 1:  # Если параметры не стандартные, переходим к стандартному распределению
        return mu0 + s * inv_f_norm(p, 0, 1, t)
    low_x, low_p = -100, 0  # Начальные границы поиска от -100 до 100
    hi_x, hi_p = 100, 1
    while hi_x - low_x > t:  # Бинарный поиск с заданной точностью
        mid_x = (low_x + hi_x) / 2
        mid_p = f_norm(mid_x)
        if mid_p < p:
            low_x, low_p = mid_x, mid_p
        elif mid_p > p:
            hi_x, hi_p = mid_x, mid_p
        else:
            break
    return mid_x

# Функция для вычисления ошибки первого рода (вероятности отклонения верной гипотезы)
# x - критическая точка, mu - среднее значение, s - стандартное отклонение
def error_of_the_first_kind(count_tiles, mu=0, s=1):
    if count_tiles >= mu:
        return 2 * (1 - f_norm(count_tiles, mu, s))
    else:
        return 2 * f_norm(count_tiles, mu, s)

# Функция для решения первой задачи
def response_for_first_task():
    count_tiles = 700 / 8  # Количество плиток, которые нужно проверить
    alpha = 0.05  # Уровень значимости
    beta = 0.8  # Требуемая мощность теста
    for n in range(1, 1000):  # Перебор различных значений n (количество часов)
        p0 = 4 / 8  # Вероятность для нулевой гипотезы
        p_alt = 3 / 8  # Вероятность для альтернативной гипотезы

        mu0 = n * p0  # Математическое ожидание для нулевой гипотезы
        mu_alt = n * p_alt  # Математическое ожидание для альтернативной гипотезы

        q0 = 1 - p0  # Дополнение вероятности для нулевой гипотезы
        sigma_0 = sqrt(n * p0 * q0)  # Стандартное отклонение для нулевой гипотезы

        q_alt = 1 - p_alt  # Дополнение вероятности для альтернативной гипотезы
        sigma_alt = sqrt(n * p_alt * q_alt)  # Стандартное отклонение для альтернативной гипотезы

        p_low = inv_f_norm(alpha / 2, mu0, sigma_0)  # Нижняя граница критической области
        p_high = 2 * mu0 - p_low  # Верхняя граница критической области
        P_value = error_of_the_first_kind(count_tiles, mu0, sigma_0)  # Вероятность ошибки первого рода

        # Вычисление мощности теста
        w = 1 - (integrate.quad(rho_norm, p_low, p_high, args=(mu_alt, sigma_alt)))[0]

        # Проверка условия на мощность теста и уровень значимости
        if P_value > alpha and w > beta:
            print(f"Мощность проверки равна {w}")
            print(f"Кол-во часов: {n}")
            break

# Запуск функции для решения первой задачи
response_for_first_task()