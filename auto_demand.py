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

# 1. Распределение значений целевой функции и построение гистограммы
def part_rent_clients(dataset,score):
    count_score = dataset.groupby(score).size()
    part_score = count_score/len(dataset)
    print('Количество объектов класса "НОВОЕ АВТО" составляет '+ str(count_score[1]))
    print('Количество объектов класса "НЕ НОВОЕ АВТО" составляет '+ str(count_score[0]))
    print('Доля класса "НОВОЕ АВТО": '+ str((part_score[1])*100)+ ' %')
    print('Доля класса "НЕ НОВОЕ АВТО": '+ str((part_score[0])*100)+ ' %')
    
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

#РАБОТАЕМ С ДАТАСЕТОМ РЕАЛЬНЫХ ЛИЗИНГОВЫХ СДЕЛОК
#import dataset
auto_clients = pd.read_csv('/home/anton/Projects/python/development/credit_demand/leasing_clid_20190603.csv', encoding = "ISO-8859-1")
#auto_clients = pd.read_csv('D:/Models/development/credit_demand/leasing_clid_20190613_rem.csv', encoding = "ISO-8859-1")

auto_clients.shape
# Посмотрим на распределение по признаку нового авто
part_rent_clients(auto_clients,'IS_LEASING')