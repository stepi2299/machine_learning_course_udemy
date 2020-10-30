import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing UCB

N = 10000  # total number of round (users in our case) remember that it is a simulation and normally this number whould be dynamic and still increasing until we would stop exploiting
d = 10  # number of add (cases)
ads_selected = []  # full list things that was rewarded during all rounds (in our case the ads which was clicked)
numbers_of_selections = [0] * d  # inicialize list of ten zeros (number of times that the thing was rewarded (Ni(n) from the algorithm)
# when one of the things will be rewarded it will increase +1 in proper place in list
sums_of_rewards = [0] * d  # the sum of rewards of the thing (Ri(n) from the algorithm)
total_reward = 0  # total reward accumulated over the rounds with the different things that was rewarded

for n in range(N):
    ad = 0  # we are starting with first thing to iterate through all and select with the highest upper confidence bound
    max_upper_bound = 0  # at the begining
    for a in range(d):
        if numbers_of_selections[a] > 0:  # here we are checking if the thing was rewarded
            average_reward = sums_of_rewards[a] / numbers_of_selections[a]
            delta_i = math.sqrt(3/2 * math.log(n+1)/ numbers_of_selections[a])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound  # here we use python trick to have certainity that our max upper bound will be updated
            ad = a
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the result
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
