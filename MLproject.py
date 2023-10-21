#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate - Price Prediction

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head


# In[4]:


housing.info


# In[5]:


housing.describe() 


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib as plt


# In[8]:


housing.hist(bins=50,figsize=(20,15))


# ## Train - Test Splitting

# In[9]:


import numpy as np

def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled=np.random.permutation(len(data))
    print(shuffled)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled[:test_set_size]
    train_indices=shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[10]:


train_set,test_set=split_train_test(housing,0.2)


# In[11]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# ## Using SK-Learn to split data 

# In[12]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


housing['CHAS'].value_counts()

#Since the 0's and 1's are'nt close to being evenly distributed we will modify the train and test sets to avoid overfitting. We perform stratified sampling
# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing ,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    


# In[15]:


strat_test_set['CHAS'].value_counts() 


# In[16]:


strat_train_set['CHAS'].value_counts()


# ## Looking for Correlations

# In[17]:


corr_matrix=housing.corr()


# In[18]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[19]:


from pandas.plotting import scatter_matrix
attributes=["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[20]:


housing.plot(kind="scatter",x="RM",y="MEDV",alpha=0.8)

#We will try to remove the outliers for us to achieve a clean data set and remove erronious values. For instance, 1 value of 5 RMs points at 50 which is the same for a value point by 9 RM which is highly unlikely. Hence we try to remove the outliers.
# ## Trying out attribute combinations

# In[21]:


housing["TAXRM"]=housing["TAX"]/housing["RM"]


# In[22]:


housing.head()


# In[23]:


housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha=0.8)


# In[24]:


housing=strat_train_set.copy()


# In[25]:


housing


# In[26]:


housing=strat_train_set.drop("MEDV",axis=1)
housing_labels=strat_train_set["MEDV"].copy()


# In[27]:


housing.drop("RM",axis=1).shape


# In[28]:


housing.describe() 


# ## Scikit-Learn Design
# Three types of objects
# 1- Estimators to estimate some parameter based on a dataset. Has a fit and transfrom method.
# Fit-> Fits the dataset and calculates internal parameters.
# 2- Transformers- takes input and returns output based on the learnings from fit(). fit_transform() fits and then transforms.
# 3- Predictors- Eg- Linear Regression model. fit() and predict() are two common functions. Also gives score() function which evaluates the predictions.

# ## Creating a pipeline

# In[29]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
imputer.fit(housing)


# In[30]:


imputer.statistics_


# In[31]:


X=imputer.transform(housing)


# In[32]:


housing_tr=pd.DataFrame(X,columns=housing.columns)


# In[33]:


housing_tr.describe()


# In[34]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


# In[ ]:





# In[35]:


housing_num_tr=my_pipeline.fit_transform(housing)


# In[36]:


housing_num_tr.shape


# ## Selecting a desired model

# In[37]:


housing_num_tr.shape


# In[ ]:





# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model=LinearRegression()
#model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[ ]:





# In[39]:


some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]


# In[40]:


prepared_data=my_pipeline.transform(some_data)


# In[41]:


model.predict(prepared_data)


# In[42]:


list(some_labels)


# ## Evaluating the Model

# In[43]:


from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels,housing_predictions)
rmse=np.sqrt(mse)


# In[44]:


mse


# We got a very high error using Linear regression(23%). Thus we will be using a different model i.e. Decision Tree regressor. This initially gave us an error of 0. Thus, overfitting. We use cross validation

# ## Cross Validation

# In[45]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)


# In[46]:


rmse_scores=np.sqrt(-scores)


# In[47]:


rmse_scores


# In[48]:


def print_scores(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard deviation:",scores.std())


# In[49]:


print_scores(rmse_scores)


# ## Saving the Model

# In[50]:


from joblib import dump, load
dump(model,'Dragon.joblib')


# ## Testing the Model on test data

# In[56]:


x_test=strat_test_set.drop("MEDV",axis=1)
y_test=strat_test_set["MEDV"].copy()
x_test_prepared=my_pipeline.transform(x_test)
final_predictions=model.predict(x_test_prepared)
final_mse=mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions,list(y_test))


# In[55]:


final_rmse


# In[58]:


prepared_data[0]


# ## Using the model

# In[59]:


from joblib import dump,load
import numpy as np
model=load('Dragon.joblib')
input=np.array([[-0.44241248,  3.18716752, -1.12581552, -0.27288841, -1.42038605,
       -0.54601796, -1.7412613 ,  2.56284386, -0.99534776, -0.57387797,
       -0.99428207,  0.43852974, -0.49833679,  0.        ]])
model.predict(input)

