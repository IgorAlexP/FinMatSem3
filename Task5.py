### Задача 5. Аналитик собрал статистические данные между ценой акции перерабатывающей компании и ценой ресурса,
# который эта компания перерабатывает: Месяц 1 2 3 4 5 6 7 8 9 10 11 12 Цена акции, $ 12,1 15,2 15,3 15,7 15,2 16,1 16,5 17,1 17,2 17,0 16,8 16,9 
# Цена ресурса, $ 115,0 119,0 121,0 130,0 131,0 150,0 155,0 172,0 174,0 168,0 161,0 159,0 
# И предположил, что цена акции зависит от цены ресурса с задержкой на 1 месяц. Определить уравнение регрессии 
# для этого предположения и сделать на его основе прогноз цены акции на 13 месяц. ###


import numpy as np
import matplotlib.pyplot as plt

# Данные
months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
stock_price = np.array([12.1, 15.2, 15.3, 15.7, 15.2, 16.1, 16.5, 17.1, 17.2, 17.0, 16.8, 16.9])
resource_price = np.array([115.0, 119.0, 121.0, 130.0, 131.0, 150.0, 155.0, 172.0, 174.0, 168.0, 161.0, 159.0])

# Создаем матрицу X с задержкой на 1 месяц для цены ресурса
X = np.column_stack((np.ones_like(months), np.roll(resource_price, 1)))

# Выполняем метод наименьших квадратов
coefficients = np.linalg.lstsq(X, stock_price, rcond=None)[0]

# Выводим уравнение регрессии
intercept, slope = coefficients
equation = f"Цена акции = {intercept:.2f} + {slope:.2f} * Цена ресурса (задержка 1 месяц)"

# Прогноз цены акции на 13 месяц
predicted_price_13th_month = intercept + slope * resource_price[-1]
print(f"Прогноз цены акции на 13 месяц: {predicted_price_13th_month:.2f} $")

# Строим график
plt.scatter(resource_price, stock_price, label='Данные')
plt.plot(resource_price, intercept + slope * resource_price, color='red', label='Уравнение регрессии')
plt.scatter(resource_price[-1], predicted_price_13th_month, color='green', label='Прогноз на 13 месяц')
plt.xlabel('Цена ресурса')
plt.ylabel('Цена акции')
plt.legend()
plt.show()

### Вывод: Прогноз цены акции на 13 месяц: 16.20 $  График в папке images ###