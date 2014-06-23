# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 22:34:05 2014

@author: arnaudlina
"""

from __future__ import division
from numpy import *
import matplotlib.pyplot as plt
from numpy import arange
import random as rnd
import sys    
import matplotlib.cm as cm

 
class AdaBoost:
 
    def __init__(self, training_set, testing_set):
        
        self.training_set = training_set
        self.testing_set  = testing_set
        
        self.weights    = []

        self.RULES      = []
        self.RULESINDEX = []
        
        self.ALPHA      = []
        
        self.LEARNER    = []
        self.LEARNERDEF = []
        
        self.ERRORTEST  = []
        self.ERRORTRAIN = []
        self.ERRORUPPER = []
        

    def get_funcerror(self, func):
        errors = array([t[1]!=func(t[0]) for t in self.training_set])
        return (errors*self.weights).sum()
        
    def set_rule(self, func):
        
        e = self.get_funcerror(func)        
        alpha = 0.5 * log((1-e)/e)
    
        for i, t in enumerate(self.training_set):
            self.weights[i] = self.weights[i]*exp(-alpha*t[1]*func(t[0]))
    
        Z = self.weights.sum()
        self.weights = self.weights / Z
        
        self.RULES.append(func)
        self.ALPHA.append(alpha)
        
        return Z
        
    def add_learner(self, func):
        self.LEARNER.append(func)
    
    def add_learner_def(self, dim, value, sign):
        self.LEARNERDEF.append((dim, value, sign))
        
    def build_stump_inf(self, dim, value) : return lambda x: 2*(x[dim] < value)-1
    
    def build_stump_sup(self, dim, value) : return lambda x: 2*(x[dim] > value)-1
        
    def add_stumps(self):
        for dim in range(len(self.training_set[0][0])):
            for t, l in self.training_set:
                self.add_learner(self.build_stump_inf(dim, t[dim]))
                self.add_learner(self.build_stump_sup(dim, t[dim]))
                self.add_learner_def(dim, t[dim], 'inf')
                self.add_learner_def(dim, t[dim], 'sup')
                
                
    def evaluate(self, data):

        hx = [self.ALPHA[i]*self.RULES[i](data) for i in range(len(self.RULES))]
        s = sum(hx)        
        c = s/sum(self.ALPHA)      
        r = sign(s)
        
        return (r, absolute(c))
        
        
    def test(self):
        
        rate=zeros(2)
        
        for (x,l) in self.testing_set:
            res = self.evaluate(x)

            if (sign(l) != sign(res[0])) :
                rate[sign(l)] = rate[sign(l)] +1 
                
        print " - test_0:" + str(rate[0]) + " test_1:"+str(rate[1]) + " total :" + str(rate[0]+rate[1]) + " over:" + str(len(self.training_set))
        return rate
                
    def train(self, iter=100, drawit=False):
        
        self.ERRORTRAIN=[]
        self.ERRORTEST=[]
        
        if len(self.LEARNER)==0:
            return False
        
        if len(self.training_set)==0:
            return False
        
        nb_class_pos = (array([1 for t,l in self.training_set if sign(l)>0])).sum()
        nb_class_neg = (array([1 for t,l in self.training_set if sign(l)<0])).sum() 
        
        self.weights = ones(len(self.training_set))
        
        for i,t in enumerate(self.training_set):
            if(sign(t[1])>0):
                self.weights[i] = 1.0 / float(2.0*nb_class_pos)
            else:
                self.weights[i] = 1.0 / float(2.0*nb_class_neg)
         
         
        for t in range(0, iter):
            
            print "Training iteration: " + str(t)
            
            err_learner = array([self.get_funcerror(cur_learner) for cur_learner in self.LEARNER])
            min_index = (where(err_learner == err_learner.min()))[0][0]        
            min_error = err_learner[min_index]
            
            if(min_error>=0.5):
                return False
            
            print " - min_error:" + str(min_error) + " @ " + str(min_index) + " learner_info:" + str(self.LEARNERDEF[min_index]) 
            
            
            Zt = self.set_rule(self.LEARNER[min_index])
            self.RULESINDEX.append(min_index)
            
            self.ERRORTRAIN.append(min_error)

            self.ERRORUPPER.append(Zt)
            
            test_res = self.test()
            self.ERRORTEST.append((test_res[0]+test_res[1])*100.0/len(self.training_set))
            
            if t>0:
                self.ERRORUPPER[t] = self.ERRORUPPER[t]*self.ERRORUPPER[t-1]
            
            if drawit:
                self.draw()
                
            #sys.stdin.read(1)
    
        return True
        
            
    def draw(self):
        pass_h_c0_x = []
        pass_h_c0_y = []
        pass_l_c0_x = []
        pass_l_c0_y = []
        pass_h_c1_x = []
        pass_h_c1_y = []
        pass_l_c1_x = []
        pass_l_c1_y = []
        fail_c0_x   = []
        fail_c0_y   = []
        fail_c1_x   = []
        fail_c1_y   = []
        
        c_level = 0.1
        
        for t, l in self.testing_set:
            res = self.evaluate(t)
            if sign(l)>0:
                if sign(res[0])>0:
                    if(res[1]>c_level):
                        pass_h_c0_x.append(t[0])
                        pass_h_c0_y.append(t[1])
                    else:
                        pass_l_c0_x.append(t[0])
                        pass_l_c0_y.append(t[1])
                else:
                    fail_c0_x.append(t[0])
                    fail_c0_y.append(t[1])
            else:
                if sign(res[0])<0:
                    if(res[1]>c_level):
                        pass_h_c1_x.append(t[0])
                        pass_h_c1_y.append(t[1])
                    else:
                        pass_l_c1_x.append(t[0])
                        pass_l_c1_y.append(t[1])
                else:
                    fail_c1_x.append(t[0])
                    fail_c1_y.append(t[1])
                
        fig, ax = plt.subplots()        
        ax.plot(pass_l_c0_x, pass_l_c0_y, 'c+', pass_l_c1_x, pass_l_c1_y, 'm+')
        ax.plot(pass_h_c0_x, pass_h_c0_y, 'g+', pass_h_c1_x, pass_h_c1_y, 'r+')        
        ax.plot(fail_c0_x, fail_c0_y, 'go', fail_c1_x, fail_c1_y, 'ro')

        fig2, ax2 = plt.subplots()
        ax2.plot(self.ERRORTRAIN, 'r-', self.ERRORUPPER, 'c-')
        ax2.set_ylabel('Error (r) Upper (c)', color='b')
        ax3 = ax2.twinx()
        ax3.plot(self.ERRORTEST, 'g-')
        ax3.set_ylabel('TestError %', color='g')
        
        
        z = [[0 for ni in range(300)] for mi in range(300)]
        l = [[0 for ni in range(300)] for mi in range(300)]
        
        for x in range(-150, 150):
            for y in range(-150, 150):
                z[150-y-1][x+150]=(self.evaluate((x,y)))[1]
                l[150-y-1][x+150]=(self.evaluate((x,y)))[0]
        
        plt.figure()
        plt.imshow(z, interpolation='bilinear', cmap=cm.jet)
        plt.title('Confidence map')
        plt.gcf()
        plt.clim()
        
        plt.figure()
        plt.imshow(l, interpolation='bilinear', cmap=cm.jet)
        plt.title('Classification map')
        plt.gcf()
        plt.clim()
        
        plt.show()
   
         
if __name__ == '__main__':
 
    train_set = []
    test_set  = []
    
    nb_data = 500;
    cur=0
    while (cur<nb_data):
        x=(rnd.randrange(-100, 100), rnd.randrange(-100, 100))
        dist = sqrt(x[0]*x[0]+x[1]*x[1])
        if (dist>50) or (dist<20):
            train_set.append((x, 1))
            cur=cur+1
            
    nb_data = 50;
    cur=0
    while (cur<nb_data):
        x=(rnd.randrange(-100, 100), rnd.randrange(-100, 100))
        dist = sqrt(x[0]*x[0]+x[1]*x[1])
        if (dist<50) and (dist>20):
            train_set.append((x, -1))
            cur=cur+1

    train_set.append(((-150,-150), 1))
    train_set.append(((150,150), 1))
    
    test_set = train_set
         
    print("--- NEW ---")
    
    m = AdaBoost(train_set, test_set)


    m.add_stumps()
        
    e=m.train(500, True)
    
    
 
