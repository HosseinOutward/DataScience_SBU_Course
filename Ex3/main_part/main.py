import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

df=pd.read_csv("src_data/genres_v2.csv", sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
f='danceability'
g='tempo'

pcntls=df[[g,f]].groupby([g]).describe(percentiles=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
print(pcntls)
sumry= pcntls[f].T
#print sumry
ordered=sorted(df[g].dropna().unique())

#set figure size and scale text
plt.rcParams['figure.figsize']=(15,10)
text_scaling=1.9
sns.set(style="whitegrid")
sns.set_context("paper", font_scale=text_scaling) 

#plot boxplot
ax=sns.boxplot(x=df[g],y=df[f],width=0.5,order=ordered, whis=[10,90],data=df, showfliers=False,color='lightblue', 
            showmeans=True,meanprops={"marker":"x","markersize":12,"markerfacecolor":"white", "markeredgecolor":"black"})

plt.axhline(0.3, color='green',linestyle='dashed', label="S%=0.3")

#this line sets the scale to logarithmic
ax.set_yscale('log')

leg= plt.legend(markerscale=1.5,bbox_to_anchor=(1.0, 0.5) )#,bbox_to_anchor=(1.0, 0.5)
#plt.title("Assay data")
plt.grid(True, which='both')
ax.scatter(x=sorted(list(sumry.columns.values)),y=sumry.loc['5%'],s=120,color='white',edgecolor='black') 
ax.scatter(x=sorted(list(sumry.columns.values)),y=sumry.loc['95%'],s=120,color='white',edgecolor='black')


#add legend image
img = plt.imread("legend.jpg")
plt.figimage(img, 1900,900, zorder=1, alpha=1)


#next line is important, select a column that has no blanks or nans as the total items are counted. 
df['value']=df['From']

vals=df.groupby([g])['value'].count()
j=vals

ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
#print ymax

#put n= values at top of plot    
x=0
for i in range(len(j)):

    plt.text(x = x , y = ymax+0.2, s = "N=\n" +str(int(j[i])),horizontalalignment='center')
    #plt.text(x = x , y = 102.75, s = "n=",horizontalalignment='center')
    x+=1





#use the section below to adjust the y axis lable format to avoid default of 10^0 etc for log scale plots.
ylabels = ['{:.1f}'.format(y) for y in ax.get_yticks()]
ax.set_yticklabels(ylabels)