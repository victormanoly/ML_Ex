import math
import pandas as pd

df=pd.read_csv('nebulosa_train.txt', header=None, sep=' ')

print(df)


for row in df.itertuples():
    for i in row:
       if i == -100:
            df=df.replace(-100, 'NaN')  #Replace every element with -100 value

#df=df.dropna()  #Drop the rows where all elements are missing and Keep the DataFrame with valid entries in the same variable
               
print(df)


