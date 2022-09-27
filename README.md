# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Худорожков Георгий Олегович
- РИ211102
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)
## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.
## Задание 1
### Написать программы Hello World на Pyton и Unity
- Скриншот google.collab.
- ![image](https://user-images.githubusercontent.com/114441283/192611408-87d4fa81-cd42-4ac7-bdf3-ae56a697aeca.png)
- Ссылка https://colab.research.google.com/drive/1jzbq7LtQqfR6u6Xf9hEfrqG6Nq9n2Rcd
- Скриншот Jupyter.
- ![image](https://user-images.githubusercontent.com/114441283/192611627-9e6a6892-4b82-4839-b090-34d613a2195a.png)
- Скриншот Unity.
-  ![image](https://user-images.githubusercontent.com/114441283/192611929-27d5d33b-0a44-4d09-98fe-75407e833a61.png)
## Задание 2
### Пошагово выполнить каждый пункт раздела "ход работы" с описанием и примерами реализации задач
Ход работы:
-1 Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и данные находились в линейной зависимости. Данные преобразуются в формат массива, чтобы их можно было вычислить напрямую при использовании умножения и сложения.

```py

In [ ]:
#Import the required modules, numpy for calculation, and Matplotlib for drawing
import numpy as np
import matplotlib.pyplot as plt
#This code is for jupyter Notebook only
%matplotlib inline

# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#Show the effect of a scatter plot
plt.scatter(x,y)

```
- Скриншот 
![image](https://user-images.githubusercontent.com/114441283/192613309-0118ab5c-32cb-4ae0-a30e-715eac5d8ab2.png)

-2 Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь среднеквадратичной ошибки. Функция оптимизации: метод градиентного спуска для нахождения частных производных w и b.
```py
def model (a, b, x):
    return a*x + b

def loss_function (a, b, x, y):
    num = len (x)
    prediction = model(a, b, x) 
    return (0.5/num) * (np.square (prediction-y)).sum()

def optimize (a, b, x, y) :
    num = len (x)
    prediction = model (a, b,x)
    
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum ())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

 
def iterate (a,b, x, y, times):
    for i in range (times):
        a,b = optimize (a,b,x,y)
    return a, b
```
- Скриншот 
![image](https://user-images.githubusercontent.com/114441283/192619842-68c0877a-3749-4b58-8374-752af35cea9f.png)

-3 Начать итерацию.
- Шаг 1. Инициализация и модель итеративной оптимизации.
```py
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a,b,x,y,1)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)
plt.scatter(x,y)
plt.plot(x,prediction)
```
- Скриншот 
![image](https://user-images.githubusercontent.com/114441283/192614424-52deaad8-ca87-4ea5-92b3-bd5008eeb1d4.png)
- Шаг 2. На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
```py
a,b = iterate (a, b, x, y, 2)
prediction=model (a, b, x)
loss = loss_function (a, b, x, y)
print (a,b, loss) 
plt.scatter (x, y)
plt.plot (x,prediction)
```
- Скриншот
 ![image](https://user-images.githubusercontent.com/114441283/192614872-8eb68da9-c65e-4fbf-9dc8-d6d39b89f4a7.png)
- Шаг 3. Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации
```py
a,b = iterate (a,b,x,y,3)
prediction=model (a, b, x)
loss = loss_function (a,b,x,y)
print (a, b, loss) 
plt.scatter (x, y)
plt.plot (x,prediction)
```
- Скриншот
![image](https://user-images.githubusercontent.com/114441283/192615266-50cf6648-3421-4102-bab1-8fe2c94645e8.png)
- Шаг 4. На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
```py
a,b = iterate (a,b,x,y,4)
prediction=model (a, b, x)
loss = loss_function (a,b,x,y)
print (a, b, loss) 
plt.scatter (x, y)
plt.plot (x,prediction)
```
- Скриншот
![image](https://user-images.githubusercontent.com/114441283/192615786-2d8806b2-ea5a-4e51-a2cd-ed1e92d19361.png)
- Шаг 5. Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации
```py
a,b = iterate (a,b,x,y,5)
prediction=model (a, b, x)
loss = loss_function (a,b,x,y)
print (a, b, loss) 
plt.scatter (x, y)
plt.plot (x,prediction)

```
- Скриншот
 ![image](https://user-images.githubusercontent.com/114441283/192616086-0027e583-7365-4b8c-bc52-902dec869ea2.png)
- Шаг 6. 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
```py
a,b = iterate (a,b,x,y,10000)
prediction=model (a, b, x)
loss = loss_function (a,b,x,y)
print (a, b, loss) 
plt.scatter (x, y)
plt.plot (x,prediction)
```
- Скриншот
![image](https://user-images.githubusercontent.com/114441283/192616337-85a862d7-e2b9-41d1-b555-8600b0a09d42.png)


## Выводы

В ходе выполненной лабораторной работы были получены начальные знания о работе в Unity, Jupyter, github и google.collab.

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
