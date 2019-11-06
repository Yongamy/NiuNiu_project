import numpy as np
import pandas as pd

df = pd.read_excel('DR007日月数据.xls')
print(df.head())
se = df.iloc[:, 2]
se = se.dropna()
df = pd.DataFrame(se)
df.to_excel('DR007日月数据.xlsx', index=False)
#new = df.drop(columns=['时间'])
#new['时间'] = df.apply(lambda x: x.时间[0:4] + '.' + x.时间[5:7], axis=1)
#new['时间'] = df.apply(lambda x: x.时间.strftime('%Y.%m'), axis=1)
#new.to_excel('DR007日月数据.xls', index=False)
