import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df = sns.load_dataset("penguins")
sns.set_style("darkgrid")
sns.jointplot(data=df, x="flipper_length_mm", y="bill_length_mm", hue= "species" )
plt.figure()

