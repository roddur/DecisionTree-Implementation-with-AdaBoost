#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decision Tree Algorithm

@author: roddur
"""

from collections import Counter
import numpy as np
import math

class DecisionTree:
    
    """
    works with integer attribute-label examples with no missing values
    
    """
    
    depth = 100
    
    class Tree:
    
        def __init__(self, value):
            self.value = value
            self.branches = []
        
        def add_branch(self, tree):
            self.branches.append(tree)
        
        def __str__(self):
            stri=str(self.value)+'('
            for i in self.branches:
                stri+=i.__str__()+','
            stri += ')'

            return stri
        
    
    def plurality_value(self, arr):
        
        cnt = Counter()
        for i in arr:
            cnt[i] += 1
            
        return cnt.most_common(1)
    
    
    def entropy(self, arr):
        
        cnt = Counter()
        for i in arr:
            cnt[i] += 1
            
        base = 0.0
        for i in cnt:
            base -= cnt[i]/arr.shape[0] * math.log2(cnt[i]/arr.shape[0])
            
        return base
    
    
    def importance(self, examples, attrs):
        
        ents = np.array([])
        for a in attrs:
            cnt = Counter()
            for i in examples:
                cnt[i[a]] += 1
            
            ent = self.entropy(examples[ :, -1])
            for i in cnt:
                ent -= (cnt[i]/examples.shape[0]) * self.entropy(examples[np.where(examples[ :, a] == i)][:, -1])
            
            ents = np.append(ents, [ent])
            
            
        return attrs[ents.argmax()]
        
    
    def train(self, features, labels, feature_desc, label_desc):
        
        data = np.c_[np.array(features), np.array(labels)]
        data_desc = feature_desc #+ label_desc
        
        self.tree = self.decision_tree_learning(data, data_desc, [i for i in range(len(feature_desc))], data, DecisionTree.depth)
        #print(self.tree)
        
    
    def decision_tree_learning(self, examples, examples_desc, attrs, parent_examples, depth):
        
        if examples.shape[0] == 0:
            return self.Tree(self.plurality_value(parent_examples[ :, -1])[0][0])
        
        if not attrs or depth == 0:
            return self.Tree(self.plurality_value(examples[ :, -1])[0][0])
        
        if self.plurality_value(examples[ :, -1])[0][1] == examples.shape[0]:
            return self.Tree(examples[ 0, -1])
        
        a = self.importance(examples, attrs)
        tree = self.Tree(a)
        
        attr_t = attrs.copy()
        attr_t.remove(a)
        
        for i in range(examples_desc[a]):
            examples_t = np.copy(examples)
            examples_t = examples_t[np.where(examples_t[ :, a] == i)]
            tree.add_branch(self.decision_tree_learning(examples_t, examples_desc, attr_t, examples, depth - 1))
            
        return tree
    
    
    def test(self, features):
        
        tree = self.tree
        while len(tree.branches) != 0:
            tree = tree.branches[features[tree.value]]
            
        return tree.value


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

dt=DecisionTree()
dt.train(ex, f, [3,3,3,2],[2])

print(dt.test([2,1,0,1]))
'''