import pandas as pd
from apyori import apriori as ap

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)  # by default firs row is header and it is skipped but here we want to take all data so we use parameter 'header'
transactions = []  # we are changing format of data because apriori require that
for i in range(7501):
    transactions.append([str(dataset.values[i, j]) for j in range(20)])  # 20 is the longest row and know we have certainty that all data in row will be include

# Training the Eclat model on the dataset
rules = ap(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# visualising the result
# displaying the first result coming from the output of the apriori function
results = list(rules)

# putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product1', 'Product2', 'Support'])

# displaying the results sorted by descending supports
print(resultsinDataFrame.nlargest(n=10, columns='Support'))

