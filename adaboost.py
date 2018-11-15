#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaboost Algorithm

Created on Wed Nov 14 20:02:03 2018

@author: roddur
"""

import numpy as np
from decision_tree import DecisionTree
import math

class AdaBoost:
    
    def resample(self, examples, weight, N):
        
        samples = np.empty((0,examples.shape[1]))
        for i in range(N):
            index = np.random.choice(examples.shape[0], p = weight)
            samples = np.append(samples, examples[index].reshape((1, examples.shape[1])), axis=0)
            
        return samples
    
    
    def normalize(self, arr):
        
        sum_ =np.sum(arr)
        for i in range(len(arr)):
            arr[i] /= sum_
            
    
    def __init__(self, examples, examples_desc, label, label_desc, learner, k):
        
        self.examples_desc = examples_desc
        self.label_desc = label_desc
        self.hypotheses_z = []
        
        data = np.c_[np.array(examples), np.array(label)]
        weight = np.full(examples.shape[0], 1/(examples.shape[0]))
        
        for i in range(k):
            sample = self.resample(data, weight, int(examples.shape[0] * 0.8))
            h = learner()
            h.train(sample[ :, 0 : sample.shape[1] - 1], sample[ :, -1], examples_desc, label_desc)
            
            error = 0.0
            count = 0
            for i in examples:
                if h.test(i) != label[count]:
                    error += weight[count]
                count += 1
            
            if error > .5:
                continue
            
            count = 0
            for i in examples:
                if h.test(i) == label[count]:
                    weight[count] *= error/(1-error)
                count += 1
                
            self.normalize(weight)
            
            self.hypotheses_z.append((h, math.log((1 - error)/error)))
            
       
    def test(self, features):
        
        tested = []
        for i in self.hypotheses_z:
            tested.append([i[0].test(features), i[1]])
            
        maj_weights = [0] * self.label_desc[0]
        
        for i in tested:
            maj_weights[int(i[0])] += i[1]
            
        max = 0
        max_i = -1
        for i in range(len(maj_weights)):
            if max < maj_weights[i]:
                max = maj_weights[i]
                max_i = i
                
        return max_i
            
'''      
ex = np.array([[0,0,0,1],
                [0,0,0,0],
                [1,0,0,1],
                [2,1,0,1],
                [2,2,1,1],
                [2,2,1,0],
                [1,2,1,0],
                [0,1,0,1],
                [0,2,1,1],
                [2,1,1,1],
                [0,1,1,0],
                [1,1,0,0],
                [1,0,1,1],
                [2,1,0,0]])
f = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])


ab = AdaBoost(ex, [3,3,3,2], f, [2], DecisionTree, 10)

print(ab.test([2,1,0,1]))
'''