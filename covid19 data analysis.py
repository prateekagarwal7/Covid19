#!/usr/bin/env python
# coding: utf-8

# In[1]:


# relation between the happiness index of the country and cases of covid 19 reported 
#done by help of eda exploratory data analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv("C:/Users/Lenovo/Desktop/covid19_Confirmed_dataset.csv")


# In[3]:


dataset.head()


# In[4]:


#cleaning the dataset by deleting the useless column
dataset.shape


# In[5]:


df = dataset.drop(["Lat","Long"],axis = 1,inplace = True)


# In[6]:


dataset.head()


# In[8]:


#aggregate the rows by country 
corona_dataset_aggregated = dataset.groupby("Country/Region").sum()# to know the summary


# In[9]:


corona_dataset_aggregated.head()# now the country and region are the index for the data frame 


# In[10]:


corona_dataset_aggregated.shape


# In[13]:


#visualise data related to a country 
corona_dataset_aggregated.loc["China"].plot()
corona_dataset_aggregated.loc["India"].plot()
plt.legend()


# In[14]:


#to determine the spread of corona virus in a country 
#calculate a good measure 
corona_dataset_aggregated.loc["China"][:3].plot()#calculate the spread of corona virus in china for first 3 days 


# In[16]:


#calculate the first derivative of the curve
corona_dataset_aggregated.loc['China'].diff().plot()#getting first derivative for each and every date
#diff() is used to determine the first derivative


# In[18]:


#maximum infection rate
corona_dataset_aggregated.loc["China"].diff().max()


# In[22]:


#maximum infection rate for all the ciuntry 
countries = list(corona_dataset_aggregated.index)# this list will storre all the country that are index 
max_infection_rate = []
for c in countries:
    max_infection_rate.append(corona_dataset_aggregated.loc[c].diff().max())
corona_dataset_aggregated["Max_infection_rate"] = max_infection_rate# adding new column to 


# In[23]:


corona_dataset_aggregated


# In[24]:


#create new dataset
corona_data = pd.DataFrame(corona_dataset_aggregated["Max_infection_rate"])


# In[25]:


corona_data


# In[26]:


happiness_report  = pd.read_csv("C:/Users/Lenovo/Desktop/worldwide_happiness_report.csv")


# In[27]:


happiness_report


# In[28]:


#drop the useless columns
useless_cols = ["Overall rank","Score","Perceptions of corruption"]


# In[29]:


happiness_report.drop(useless_cols,axis = 1, inplace = True)
happiness_report.head()


# In[31]:


happiness_report.set_index("Country or region", inplace = True)


# In[32]:


happiness_report.head()


# In[34]:


#join the dataset
corona_data.shape


# In[35]:


data = corona_data.join(happiness_report,how = "inner")# inner is used to match the rows between the 2 column
data


# In[36]:


# creating the corelation matrix between these factors 
data.corr()


# In[39]:


# daigonal values are one becpz these are correlation with itself 
#ploting the graph for thee data 
# ploting gdp per capita nd max infection rate 
x = data["GDP per capita"]
y = data["Max_infection_rate"]
sns.scatterplot(x,np.log(y))#taking the log value so that the scale is maintained 



# In[40]:


sns.regplot(x, np.log(y))# ploting the regression line 
# log scaling si the type of feature scaling


# In[41]:


x = data["Social support"]
y = data["Max_infection_rate"]
sns.scatterplot(x,np.log(y))


# In[42]:


sns.regplot(x, np.log(y))


# In[43]:


x = data["Healthy life expectancy"]
y = data["Max_infection_rate"]
sns.scatterplot(x,np.log(y))


# In[44]:


sns.regplot(x, np.log(y))


# In[45]:


x = data["Freedom to make life choices"]
y = data["Max_infection_rate"]
sns.scatterplot(x,np.log(y))


# In[46]:


#random forest doesn't support the null value 
#rescaling the attribute 
#gradient descent is useful that in regression 
#we can rescale the data using skykitlearn
#binarise the data 0,1 value above threshold is 1 
#useful in feature indicating


# standardization 
#mean = 0,
#std = 1

#feature scaling for standarisation  


# In[47]:


'''from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range(0,1))# value is between 0 and 1 

#scaled feature
x_after_min_max_scaler = min_max_scaler.fit_transform(x)

print('after min max scaler \n',x_after_min_max_scaler)'''


# In[ ]:


#standardization mean  = 0, std = 1 
#standardization  = preprocessing.standardScaler()


#scaled feature 
#x_after_standardization = Standarization.fit_transform(x)



#one end hotencoding means providing the value of one  either 0 or 1 
#checking for the label count data['remark'].value_counts()


#one_hot_encoding_data -= pd.get_dummies(data, colimn = ['remark','gender'])# all the different value are marked in0 or 1 

