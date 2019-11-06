import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./residue/无.xlsx')

sns.catplot(x="class", y="Residue", kind = "box", data=df)
plt.show()
