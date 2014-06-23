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

 
class AdaBoost:
 
    def __init__(self, training_set):
        self.training_set = training_set
        self.N = len(self.training_set)
        self.weights = ones(self.N)/self.N
        self.RULES      = []
        self.RULEDEF    = []
        self.RULEINDEX  = []
        self.ALPHA      = []
        self.LEARNER    = []
        self.ERROR      = []
        self.ERROREVAL  = []
 
    def set_rule(self, func, test=False):
        
        errors = array([t[1]!=func(t[0]) for t in self.training_set])
        e = (errors*self.weights).sum()
        
        if test: return e
        alpha = 0.5 * log((1-e)/e)
        print 'e=%.10f a=%.10f'%(e, alpha)
        w = zeros(self.N)
        for i in range(self.N):
            if errors[i] == 1: w[i] = self.weights[i] * exp(alpha)
            else: w[i] = self.weights[i] * exp(-alpha)
        self.weights = w / w.sum()
        self.RULES.append(func)
        self.ALPHA.append(alpha)
        return e
        
    def add_learner(self, func):
        self.LEARNER.append(func)
        
    def partitionInf(self, i, n) : return lambda x: 2*(x[i] < n)-1
    def partitionSup(self, i, n) : return lambda x: 2*(x[i] > n)-1

        
    def add_partition_learner(self, index, min, max, step):
        for kk in arange(min, max, step):
            m.add_learner(self.partitionInf(index, kk))
            m.add_learner(self.partitionSup(index, kk))
            self.RULEDEF.append((index,kk,0))
            self.RULEDEF.append((index,kk,1))
            
    def draw(self):
        wgreenx = []
        wgreeny = []
        wredx   = []
        wredy   = []
        fgreenx = []
        fgreeny = []
        fredx   = []
        fredy   = []
        
        for t, l in self.training_set:
            hx = [self.ALPHA[i]*self.RULES[i](t) for i in range(len(self.RULES))]
            if (sign(l) == sign(sum(hx))):
                if l==1:
                    wgreenx.append(t[0])
                    wgreeny.append(t[1])
                else:
                    wredx.append(t[0])
                    wredy.append(t[1])
            else:
                if l==1:
                    fgreenx.append(t[0])
                    fgreeny.append(t[1])
                else:
                    fredx.append(t[0])
                    fredy.append(t[1])
                
        fig, ax = plt.subplots()
        ax.plot(wgreenx, wgreeny, 'g+', wredx, wredy, 'r+')
        ax.plot(fgreenx, fgreeny, 'go', fredx, fredy, 'ro')
        
        for idx in self.RULEINDEX:
            r = self.RULEDEF[idx]
            xx=[]
            yy=[]
            if r[0]==0:
                xx=[r[1] for x in range(200)]
                yy=range(-100, 100)
            else:
                yy=[r[1] for x in range(200)]
                xx=range(-100, 100)
                
            if(r[2]==0):
                ax.plot(xx,yy,'y')
            else:
                ax.plot(xx,yy,'c')

        fig2, ax2 = plt.subplots()
        ax2.plot(self.ERROR, 'r-')
        ax2.set_ylabel('IterError', color='r')
        ax3 = ax2.twinx()
        ax3.plot(self.ERROREVAL, 'g-')
        ax3.set_ylabel('EvalError', color='g')
        
        plt.show()
        
        
    def train(self, iter):
        self.ERROR=[]
        self.ERROREVAL=[]
        
        if len(self.LEARNER)<=0:
            return False
         
        for i in range(0, iter):
            print "adaboost iter: " + str(i)
            
            err = 1.0
            best_learner=[]
            bestindex=0
            curbesterr=0.5
            
            for index, curlearner in enumerate(self.LEARNER):
                
                print self.RULEDEF[index]
                
                curerr = self.set_rule(curlearner, True)
                
                print "  -> err = " + str(curerr)
                
                if (curerr<curbesterr):
                    curbesterr=curerr
                
                if (curerr<err):
                    best_learner=curlearner
                    err=curerr
                    bestindex=index

            print "best error is " + str(curbesterr)
            if curbesterr>=0.499990:
                return self.ERROR
            
            print "selected leaner is " + str(bestindex)
            print "selected rule is " + str(self.RULEDEF[bestindex])
            self.RULEINDEX.append(bestindex)
            newerr = self.set_rule(best_learner, False)
            self.ERROR.append(newerr*100)
            
            self.ERROREVAL.append(self.evaluate())
            
            self.draw()
            
            #sys.stdin.read(1)
            
            #self.evaluate()
        
        return self.evaluate()
                
 
    def evaluate(self):
        NR = len(self.RULES)
        f=0
        g=0
        for (x,l) in self.training_set:
            hx = [self.ALPHA[i]*self.RULES[i](x) for i in range(NR)]
            #fx = sum(hx)/sum(self.ALPHA)
            if (sign(l) != sign(sum(hx))) and sign(l)==1:
                f=f+1
            if (sign(l) != sign(sum(hx))) and sign(l)==0:
                g=g+1
        print "eval is " + str(f) + " and "+ str(g) 
        rate = (f+g)*100.0/len(self.training_set)
        print "rate " + str(rate)
        return rate
        
         
if __name__ == '__main__':
 
    examples = []
    
    tr = [(rnd.randrange(-100, 100), rnd.randrange(-100, 100)) for x in range(1000)]
    nb_data = 1000;
    cur=0
    while (cur<nb_data):
        x=(rnd.randrange(-100, 100), rnd.randrange(-100, 100))
        if not((x[0]>-30) and (x[0]<30) and (x[1]>-10) and (x[1]<10)):
            examples.append((x, 1))
            cur=cur+1
            
    cur=0
    while (cur<nb_data):
        x=(rnd.randrange(-30, 30), rnd.randrange(-30, 30))
        if (x[0]>-30) and (x[0]<30) and (x[1]>-10) and (x[1]<10):
            examples.append((x, -1))
            cur=cur+1
         
    print("NEW")
    m = AdaBoost(examples)
    print("   -add learners")

    m.add_partition_learner(0, -100, 100, 5)
    m.add_partition_learner(1, -100, 100, 5)
    
        
    #m.add_learner(lambda x: 2*(x[0] < 1.5)-1)
    #m.add_learner(lambda x: 2*(x[0] < 4.5)-1)
    #m.add_learner(lambda x: 2*(x[1] > 5)-1)
        
    print("   -train")
    e=m.train(100)
    plt.plot(e)
    
    m.draw()
        
    print("   -evaluate")
    m.evaluate()
    
    
 
