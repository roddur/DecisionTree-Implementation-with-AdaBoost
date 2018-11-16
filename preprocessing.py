#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 08:15:37 2018

@author: roddur
"""

import math
import pandas as pd
from collections import Counter

class Preprocessing:
    
    def read_file(filename):
        data = pd.read_csv(filename)
        return data
    
    
    def drop_column(data, column_name):
        return data.drop(columns=column_name)
        
    
    def integer_conversion(data, column_name, column_category):
        data[column_name] = data[column_name].astype("category", categories=column_category).cat.codes
        return data
    
    
    def entropy_from_counter(cnt):
        base = 0.0
        
        tot = sum(cnt)
        for i in cnt:
            if cnt[i] > 0:
                base += cnt[i]/tot * math.log2(cnt[i]/tot)

        return base
    
    
    def binarization(data, column_name):
        
        data.sort_values(by=[column_name])
        labels = data.iloc[:, -1]
        right = labels.value_counts().to_dict()
        right = Counter(right)
        left = Counter()
        
        count = 0
        best = -1
        best_ent = math.inf
        
        for i in labels:
            left[i] += 1
            right[i] -= 1
            count +=1
            
            entropy = (count/labels.size)*Preprocessing.entropy_from_counter(left) + \
            ((labels.size - count)/count) * Preprocessing.entropy_from_counter(right)
            
            if best_ent > entropy:
                best_ent = entropy
                best = count
                
        threshold = (data.loc[best - 1, column_name] + data.loc[best, column_name])/2
        print(data)
        data[column_name] = (data[column_name] > threshold) * 1
        
        return data

        
data = Preprocessing.read_file('telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = Preprocessing.drop_column("customerID")
data.fillna(data.mode().iloc[0])
data = Preprocessing.integer_conversion(data, "gender", ['Female', 'Male'])
data = Preprocessing.integer_conversion(data, "Partner", ['Yes', 'No'])

data = Preprocessing.integer_conversion(data, "Churn", ['Yes', 'No'])
Preprocessing.binarization(data, "MonthlyCharges") 
