#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:11:56 2018

@author: dianabursac
"""

from nlib import Canvas, mean, sd, MCEngine
import csv
from math import exp, log, sqrt, erf
import random

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

def read_csv(filename = 'accidents.csv'):
    '''this function reads the accidents file and returns
    the vector of losses, the vector od days when the accidents happened,
    and the vector of time intervals between the accidents per each plant over
    the 4 years period'''
    dayA=[]
    dayB=[]
    lossA=[]
    lossB=[]
    time_intervals_A=[]
    time_intervals_B=[]
    previousA = 0.0
    previousB = 0.0
    with open (filename) as myfile:
        reader = csv.reader(myfile)
        for row in reader:
            if row[0] == 'A':
                dayA.append(row[1])
                lossA.append(row[2])
                dayA = map(float, dayA)
                lossA = map(float, lossA)
                time_intervals_A.append(float(row[1])-previousA)
                previousA = float(row[1])
            elif row[0] =='B':
                dayB.append(row[1])
                lossB.append(row[2])
                dayB = map(float, dayB)
                lossB = map(float, lossB)
                time_intervals_B.append(float(row[1])-previousB)
                previousB = float(row[1])
    return dayA, lossA, dayB, lossB, time_intervals_A, time_intervals_B

dayA, lossA, dayB, lossB, time_intervals_A, time_intervals_B = read_csv(filename = 'accidents.csv')


# CHAPTER 1.2

def avg_loss(day,loss):
    '''this function calculates the average loss per year'''
    avg_loss_year = sum(loss)/4
    return avg_loss_year


def avg_accidents(day,loss):
    '''this function calculates the average number of accidents per year'''
    avg_accidents_year = float(len(day))/4 
    return avg_accidents_year

def avg_loss_accident(day, loss):
    '''this function calculates the average loss per accident '''
    avg_loss_per_accident = sum(loss) / len(day)
    return avg_loss_per_accident 

avg_lossA_year = avg_loss(dayA, lossA)
avg_lossB_year = avg_loss(dayB, lossB)
avg_accidentsA_year = avg_accidents(dayA,lossA)
avg_accidentsB_year = avg_accidents(dayB,lossB)
avg_lossA_per_accident = avg_loss_accident(dayA, lossA)
avg_lossB_per_accident = avg_loss_accident(dayB, lossB)

def get_parameters(loss):
    '''this function returns mean and sigma of log loss'''
    loss_log = []
    for value in loss:
        value1 = log(value)
        loss_log.append(value1)
    mu = mean(loss_log)
    sigma = sd(loss_log)
    return loss_log, mu, sigma
    
lossA_log, muA, sigmaA = get_parameters(lossA)
lossB_log, muB, sigmaB = get_parameters(lossB)


lambA = avg_accidentsA_year
lambB = avg_accidentsB_year


# DATA VISUALISATION

# loss histogram of the original data set  
Canvas(title='Histogram of the losses in plant A', xlab='values for the losses in plant A', ylab='frequency').hist(lossA).save('lossA_hist.png')
Canvas(title='Histogram of the losses in plant B', xlab='values for the losses in plant B', ylab='frequency').hist(lossB).save('lossB_hist.png')


# time intervals histogram of the original data set
Canvas(title='Histogram of the time intervals in plant A',xlab='time intervals in plant A',ylab='frequency').hist(time_intervals_A).save('time_intervals_A.png')
Canvas(title='Histogram of the time intervals in plant B',xlab='time intervals in plant B',ylab='frequency').hist(time_intervals_B).save('time_intervals_B.png')


# Scatter Real data losses in plant A/B over the 4 years period (loss vs time)
points_T_L_A = [(dayA[i],lossA[i]) for i in range(len(lossA))]
points_TA = [x[0] for x in points_T_L_A]
points_LA = [x[1] for x in points_T_L_A]
plt.scatter(points_TA, points_LA)
plt.title("Real data losses in plant A over the 4 years period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()

points_T_L_B = [(dayB[i],lossB[i]) for i in range(len(lossB))]
points_TB = [x[0] for x in points_T_L_B]
points_LB = [x[1] for x in points_T_L_B]
plt.scatter(points_TB, points_LB)
plt.title("Real data losses in plant B over the 4 years period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()

def simulate_once(lamb, mu, sigma): # simulate one day
    '''this function returns accumulated simulated losses, simulated losses
    and simulated time/days'''
    t = 0.0 # start
    accum_loss_sim=[]
    accum_loss = 0.0
    time_sim = []
    loss_sim = []
    while True: # t is 1 year
        delta = random.expovariate(lamb) # time interval between accidents
        t = t+delta
        if t>1: break
        loss = random.lognormvariate(mu,sigma)
        accum_loss = loss+accum_loss
        loss_sim.append(loss)
        time_sim.append(t*365)
        accum_loss_sim.append(accum_loss)
    return accum_loss_sim, loss_sim, time_sim

accum_lossA_sim, lossA_sim, timeA_sim = simulate_once(lambA, muA, sigmaA)
accum_lossB_sim, lossB_sim, timeB_sim = simulate_once(lambB, muB, sigmaB)


# Scatter Simulated losses in plant A/B  over 1 year period (loss vs time)
points_T_L_A_sim = [(timeA_sim[i],lossA_sim[i]) for i in range(len(lossA_sim))] # time, loss
points_TA_sim = [x[0] for x in points_T_L_A_sim] # time A
points_LA_sim = [x[1] for x in points_T_L_A_sim] # loss A
points_T_L_B_sim = [(timeB_sim[i],lossB_sim[i]) for i in range(len(lossB_sim))] # time, loss
points_TB_sim = [x[0] for x in points_T_L_B_sim] # time B
points_LB_sim = [x[1] for x in points_T_L_B_sim] # loss B
plt.scatter(points_TA_sim, points_LA_sim)
plt.title("Simulated losses in plant A  over 1 year period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()

plt.scatter(points_TB_sim, points_LB_sim)
plt.title("Simulated losses in plant B  over 1 year period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()

# Scatter Simulated cumulative losses in plant A/B  over 1 year period (cumulative loss vs time
points_T_L_A_sim_accum = [(timeA_sim[i],accum_lossA_sim[i]) for i in range(len(accum_lossA_sim))] # time, loss
points_TA_sim_accum = [x[0] for x in points_T_L_A_sim_accum] # time A
points_LA_sim_accum = [x[1] for x in points_T_L_A_sim_accum] # loss A
points_T_L_B_sim_accum = [(timeB_sim[i],accum_lossB_sim[i]) for i in range(len(accum_lossB_sim))] # time, loss
points_TB_sim_accum = [x[0] for x in points_T_L_B_sim_accum] # time B
points_LB_sim_accum = [x[1] for x in points_T_L_B_sim_accum] # loss B
plt.scatter(points_TA_sim_accum, points_LA_sim_accum)
plt.title("Simulated cumulative losses in plant A  over 1 year period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()
    
plt.scatter(points_TB_sim_accum, points_LB_sim_accum)
plt.title("Simulated cumulative losses in plant B  over 1 year period ")
plt.xlabel("days")
plt.ylabel("losses")
plt.show()   

# CHAPTER 1.3

def F(v):
    v.sort()
    n = len(v)
    data = []
    for k in range(n):
        point = (v[k], float(k+1)/n)
        data.append(point)
    return data

def F_exponential(x, lamb):
    return 1.0 - (exp(-float(lamb/365*x)))

def F_lognormal(x, mu, sigma):
    return 0.5 + 0.5*erf((log(x) - mu)/(sigma*sqrt(2.0)))

# CDF time intervals: the original data vs model plant A

points_time_A = F(time_intervals_A) # original distribution based on the given data set
points_time_A_sim = [(x,F_exponential(x, lambA)) for (x,y) in points_time_A] # this is what we have created
Canvas(title='CDF comparison - exp. distibution of the time intervals in plant A',xlab='time intervals in plant A',ylab='CDF_A').plot(points_time_A, color ='blue', legend = 'original distribution plant A').plot(points_time_A_sim, color='red', legend = 'created distribution A').save('time_intervals_comparison_A.png')


# CDF time intervals: original data vs model plant B

points_time_B = F(time_intervals_B) # original distribution based on the given data set
points_time_B_sim = [(x,F_exponential(x, lambB)) for (x,y) in points_time_B] # this is waht we have created
Canvas(title='CDF comparison - exp. distibution of the time intervals in plant B',xlab='time intervals in plant B',ylab='CDF_B').plot(points_time_B, color ='blue', legend = 'original distribution plant B').plot(points_time_B_sim, color='red', legend = 'created distribution B').save('time_intervals_comparison_B.png')


# CDF loss: original data vs modeled plant A

points_loss_A = F(lossA) # original distribution based on the given data set
points_loss_A_sim = [(x,F_lognormal(x, muA, sigmaA)) for (x,y) in points_loss_A] # this is waht we have created
Canvas(title='CDF comparison - lognormal. distibution of the losses in plant A',xlab='losses in plant A',ylab='CDF_A').plot(points_loss_A, color ='blue', legend = 'original distribution plant A').plot(points_loss_A_sim, color='red', legend = 'created distribution A').save('loss_comparison_A.png')

# CDF loss: original data vs modeled plant B

points_loss_B = F(lossB) # original distribution based on the given data set
points_loss_B_sim = [(x,F_lognormal(x, muB, sigmaB)) for (x,y) in points_loss_B] # this is waht we have created
Canvas(title='CDF comparison - lognormal. distibution of the losses in plant B',xlab='losses in plant B',ylab='CDF_B').plot(points_loss_B, color ='blue', legend = 'original distribution plant B').plot(points_loss_B_sim, color='red', legend = 'created distribution B').save('loss_comparison_B.png')

# Monte Carlo simulation CHAPTER 1.5

class ServerSimulator(MCEngine):
    def __init__(self, lamb, mu, sigma):
        self.lamb = lamb 
        self.mu = mu
        self.sigma = sigma
    def simulate_once(self): # we give our own simulate once, simulate_many is the same
        lamb = self.lamb
        mu = self.mu
        sigma = self.sigma
        total_loss = 0.0
        max_time = 1 # we simulate 1 year
        t = 0.0 # start of simulation
        total_loss = 0.0
        while True: # t is 1 year
            delta = random.expovariate(lamb) # time interval between accidents
            t = t+delta
            if t>max_time: break
            loss = random.lognormvariate(mu,sigma)
            total_loss = loss+total_loss
        return total_loss
    
simA = ServerSimulator(lambA, muA, sigmaA)
simB = ServerSimulator(lambB, muB, sigmaB)


muA_minus, mu_A, muA_plus = simA.simulate_many(ap =0, rp=0.1, ns = 1000) # rp = 10%
muB_minus, mu_B, muB_plus = simB.simulate_many(ap = 0, rp=0.1, ns =1000)
dmuA = (muA_plus-muA_minus)/2 # this tell us how much I can trust mu, upper estimate-lower estimate
dmuB = (muB_plus-muB_minus)/2

print mu_A, dmuA
print mu_B, dmuB

historyA = simA.history[:1000]
historyB = simB.history[:1000]

Canvas(title='Approximation of the average yearly loss in plant A', xlab='MC step', ylab='average yearly loss plant A').plot(historyA).errorbar(historyA).save('lossA_avg_convergence.png')
Canvas(title='Approximation of the average yearly loss in plant B', xlab='MC step', ylab='average yearly loss plant B').plot(historyB).errorbar(historyB).save('lossB_avg_convergence.png') 

Canvas(title='Histogram of the simulated losses in plant A', xlab='values for the losses in plant A', ylab='frequency').hist(lossA_sim).save('lossA_sim_hist.png')
Canvas(title='Histogram of the simulated losses in plant B', xlab='values for the losses in plant B', ylab='frequency').hist(lossB_sim).save('lossB_sim_hist.png')

# BUDGET PLANNING

vA = []
vB = []
for i in range(100):
    muA_minus, mu_A, muA_plus = simA.simulate_many(ap =0, rp=0.1, ns = 1000)
    muB_minus, mu_B, muB_plus = simB.simulate_many(ap =0, rp=0.1, ns = 1000)
    vA.append(mu_A)
    vB.append(mu_B)
vA.sort()
vB.sort()
print vA, vA[90]
print vB, vB[90]

Canvas(title='Histogram of 100 simulated loss scenarios  in plant A', xlab='average yearly loss plant A', ylab='frequency') .hist(vA).save('vA_100scenarios.png')
Canvas(title='Histogram of 100 simulated loss scenarios  in plant B', xlab='average yearly loss plant B', ylab='frequency') .hist(vB).save('vB_100scenarios.png')    