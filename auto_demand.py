#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 23:55:52 2019

@author: anton
"""

# import librarias
import time
import numpy as np
import pandas as pd
# Matplotlib visualization
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
# Set default font size
plt.rc("font", size=14)
# Seaborn for visualization
from IPython.core.pylabtools import figsize
import seaborn as sns 
sns.set(font_scale = 2)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# 1. Распределение значений целевой функции и построение гистограммы
def part_rent_clients(dataset,score):
    count_score = dataset.groupby(score).size()
    part_score = count_score/len(dataset)
    print('Количество объектов класса "КЛИЕНТЫ С ЛИЗИНГОМ" составляет '+ str(count_score[1]))
    print('Количество объектов класса "КЛИЕНТЫ БЕЗ ЛИЗИНГА" составляет '+ str(count_score[0]))
    print('Доля класса "КЛИЕНТЫ С ЛИЗИНГОМ": '+ str((part_score[1])*100)+ ' %')
    print('Доля класса "КЛИЕНТЫ БЕЗ ЛИЗИНГА": '+ str((part_score[0])*100)+ ' %')
    
    sns.countplot(x=score,data=dataset,palette='hls')
    plt.show

# 2. Построение гистограммы категориальных признаков в разрезе классов 
def categorical_features_hist(dataset,feature,base):
    return sns.catplot(x=feature,
               hue=feature,
               col=base,
               data=dataset,
               kind='count',
               height=4,
               aspect=.7)
    
# 3. Построение распределения вещественного признака в разрезе классов
def numerical_features_distrib(dataset,feature,base):
    fig, axs = plt.subplots(1, figsize = (15,5))
    sns.set(color_codes=True)
    sns.kdeplot(dataset[feature][dataset[base] == 1],
                    color = "blue",
                    label = str(feature)+" for "+str(base)+" = 1")
    sns.kdeplot(dataset[feature][dataset[base] == 0],
                    color = "orange",
                    label = str(feature)+" for "+str(base)+" = 0")

# 4. Анализ заполнения признаков данными
def missing_values_table(dataset):
        # Общее число отсутствующих данных        
        mis_val = dataset.isnull().sum()
        # процент отсутствующих данных
        mis_val_percent = 100 * dataset.isnull().sum() / len(dataset)
        # Создание таблицы с результатами двух предыдущих операторов
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Переименование полей
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Сортировка таблицы по % отсутвующих значений по убыванию
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        # Возврат датасета с необходимой информацией
        return mis_val_table_ren_columns
    
# 5. Статистика по типам данных признаков
def dataset_params(dataset,score):
    dataset = dataset.drop(columns = [score])
    dtypes_list = pd.unique(dataset.dtypes)
    count_features = 0
    i=0
    for i in range(len(dtypes_list)):
        dt = str(dtypes_list[i])
        dt_list = list(dataset.select_dtypes(include=[dt]).columns)
        count_features += len(dt_list)
        print('- '+str(dt)+': '+str(len(dt_list)))
        i += 1
    print('The total number of predictors is '+str(count_features))

# 6. Избавление от мультиколлинеарности
def remove_collinear_features(dataset, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between REGULAR_CUSTOMER
    y = dataset['IS_LEASING']
    x = dataset.drop(columns = ['IS_LEASING'])
    
    # Calculate the correlation matrix
    corr_matrix = dataset.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    dataset = dataset.drop(columns = drops)

    dataset['IS_LEASING'] = y
               
    return dataset

# 7. Подсчет разницы между мат ожиданиями
def correl_calc(dataset):
    mean = np.array(dataset.dropna()).mean()
    std = np.array(dataset.dropna()).std()
    dataset.fillna(mean, inplace = True)
    dataset = dataset.apply(lambda x: (x - mean) / std)
    e_x1 = dataset[numeric_features.IS_LEASING == 1].mean()
    e_x2 = dataset[numeric_features.IS_LEASING == 0].mean()
    return (e_x1 - e_x2)

# 8. Построение распределения вещественных признаков в разрезе классов
def numerical_features_distrib(dataset,features_list):
    columns = features_list.Feature.iloc[:20]
    fig, axs = plt.subplots(20, figsize = (15,50))
    sns.set(color_codes=True)
    for ax, column in zip(axs, columns):
        sns.kdeplot(dataset[column][dataset['IS_LEASING'] == 1],
                    ax = ax,
                    color = "blue",
                    label = str(column)+" for IS_LEASING = 1")
        sns.kdeplot(dataset[column][dataset['IS_LEASING'] == 0],
                    ax = ax,
                    color = "orange",
                    label = str(column)+" for IS_LEASING = 0")

# 9. Построение гистограммы категориальных признаков в разрезе классов 
def categorical_features_hist(dataset,feature):
    return sns.catplot(x=feature,
               hue=feature,
               col='IS_LEASING',
               data=dataset,
               kind='count',
               height=4,
               aspect=.7)
#==============================================================================
# 1. DATA CLEANING AND FORMATTING
#==============================================================================
#import dataset
auto_clients = pd.read_csv('/home/anton/Projects/python/development/credit_demand/auto_demand_clid_20190613.csv', encoding = "ISO-8859-1")
#auto_clients = pd.read_csv('D:/Models/development/credit_demand/auto_demand_clid_20190613.csv', encoding = "ISO-8859-1")

auto_clients.shape
auto_clients.head(3)

# Посмотрим на распределение по признаку нового авто
part_rent_clients(auto_clients,'NEW_AUTO')


# Посмотрим на распределение категориальной величины - ГРУППА ПО ДОХОДУ
categorical_features_hist(auto_clients,'INCOME_GROUP_TYPE','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ ГРУППЫ ПО ДОХОДУ
categorical_features_hist(auto_clients,'DELTA_INCOME_GROUP_TYPE','NEW_AUTO')

# Посмотрим на распределение вещественной величины - УРОВЕНЬ ТРАТ
numerical_features_distrib(auto_clients,'TRAN','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ УРОВНЯ ТРАТ
categorical_features_hist(auto_clients,'DELTA_TRAN','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ ОКПО ПРЕДПРИЯТИЯ
categorical_features_hist(auto_clients,'DELTA_CLID_WORK_OKPO','NEW_AUTO')

# Посмотрим на распределение категориальной величины - СФЕРА ЗАНЯТОСТИ
categorical_features_hist(auto_clients,'CL_INDUSTR','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ СФЕРЫ ЗАНЯТОСТИ
categorical_features_hist(auto_clients,'DELTA_CL_INDUSTR','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ МЕСТА ПРОЖИВАНИЯ
categorical_features_hist(auto_clients,'DELTA_ADDR_CITY_ID_REAL','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ КОЛИЧЕСТВА ДЕТЕЙ ДО 16-и ЛЕТ
categorical_features_hist(auto_clients,'DELTA_QTY_CHLD_16','NEW_AUTO')

# Посмотрим на распределение категориальной величины - ИЗМЕНЕНИЕ В СЕМЕЙНОМ ПОЛОЖЕНИИ
categorical_features_hist(auto_clients,'DELTA_CL_FAM_ST','NEW_AUTO')

# =============================================================================
# РАБОТАЕМ С ДАТАСЕТОМ РЕАЛЬНЫХ ЛИЗИНГОВЫХ СДЕЛОК
# =============================================================================
#import dataset
auto_clients = pd.read_csv('/home/anton/Projects/python/development/credit_demand/leasing_clid_20190613_rem.csv', low_memory=False, encoding = "ISO-8859-1")
#auto_clients = pd.read_csv('D:/Models/development/credit_demand/leasing_clid_20190613_rem.csv', low_memory=False, encoding = "ISO-8859-1")

auto_clients.shape

# Посмотрим на распределение по признаку нового авто
part_rent_clients(auto_clients,'IS_LEASING')

# Удалим ненужные поля ай-ди клиентов
auto_clients = auto_clients.drop(['REP_CLID','CLID_CRM','CLID_TRAN'], axis=1)

# Заменим все значиения "Not Available" на np.nan
auto_clients = auto_clients.replace({'Not Available': np.nan})

# Удалим из датасета те поля, в которых заполнение менее 50%
missing_features = missing_values_table(auto_clients.drop(columns = ['IS_LEASING']))
missing_columns = list(missing_features[missing_features['% of Total Values'] > 50.0].index)
auto_clients = auto_clients.drop(list(missing_columns), axis = 1)

auto_clients.shape

# Посмотрим на распределение признаков по типам данных
dataset_params(auto_clients,'IS_LEASING')

# разделим признаки на количественные и вещественные
numeric_features = auto_clients.select_dtypes(include = [np.number])
numeric_features.shape

categorical_features = auto_clients.select_dtypes(include=[np.object])
categorical_features.shape

# Все количественные признаки проверим на мультиколлинеарность
numeric_features = remove_collinear_features(numeric_features, 0.6)
numeric_features.shape
dataset_params(numeric_features,'IS_LEASING')

# Обработаем категориальные признаки с помощью LabelEncoder
labelencoder = LabelEncoder()
z = len(categorical_features.columns)
for x in range(0,z):
    categorical_features.iloc[:,x] = labelencoder.fit_transform(categorical_features.iloc[:,x].values.astype(str))

categorical_features.shape

# соединяем категориальные и количественные признаки
features = pd.concat([numeric_features, categorical_features], axis = 1)
features.shape

# =============================================================================
# 2. FEATURE ENGINEERING AND SELECTION
# =============================================================================
corr_columns = list(numeric_features.drop(columns = ['IS_LEASING']).columns)
corr_values = []
nan_values = []

for (i, column) in enumerate(corr_columns):
    value = correl_calc(features[column])
    if np.isnan(value):
        nan_values.append(column)
    else:
        corr_values.append((column,np.abs(value)))

# для удобства из списка (corr_values) создадим dataframe 'corr_data':
corr_data = pd.DataFrame(corr_values, columns = ['Feature' , 'corr_value'])

# отсортируем и выведем топ-50 признаков:
sort_corr_data = corr_data.sort_values(by=['corr_value'], ascending=False)
top50_sort_corr_data = sort_corr_data[:50]
top50_sort_corr_data

# =============================================================================
# Посмотрим на влияние вещественных признаков на целевую переменную
# =============================================================================
number_corr_columns = list(numeric_features.drop(columns = ['IS_LEASING']).columns)
number_corr_values = []
number_nan_values = []

for (i, column) in enumerate(number_corr_columns):
    value = correl_calc(numeric_features[column])
    if np.isnan(value):
        number_nan_values.append(column)
    else:
        number_corr_values.append((column,np.abs(value)))

# для удобства из списка (corr_values) создадим dataframe 'corr_data':
number_corr_data = pd.DataFrame(number_corr_values, columns = ['Feature' , 'corr_value'])

# отсортируем и выведем топ-25 вещественных признаков:
sort_number_corr_data = number_corr_data.sort_values(by=['corr_value'], ascending=False)
top25_number_sort_corr_data = sort_number_corr_data[:25]
top25_number_sort_corr_data

# для вышеприведенных 25 вещественных признаков построим распределение в разрезе классов
numerical_features_distrib(features,top20_number_sort_corr_data)

# =============================================================================
# Посмотрим на влияние категориальных признаков на целевую переменную
# =============================================================================
object_corr_values = []
object_nan_values = []
object_corr_columns = list(categorical_features.columns)

for (i, column) in enumerate(object_corr_columns):
    value = correl_calc(categorical_features[column])
    if np.isnan(value):
        object_nan_values.append(column)
    else:
        object_corr_values.append((column,np.abs(value)))

# для удобства из списка (corr_values) создадим dataframe 'corr_data':
object_corr_data = pd.DataFrame(object_corr_values, columns = ['Feature' , 'corr_value'])

# отсортируем и выведем топ-25 категориальных признаков:
sort_object_corr_data = object_corr_data.sort_values(by=['corr_value'], ascending=False)
top20_object_sort_corr_data = sort_object_corr_data[:25]
top20_object_sort_corr_data