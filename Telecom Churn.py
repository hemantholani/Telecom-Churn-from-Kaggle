#!/usr/bin/env python
# coding: utf-8

# # INSAID Hiring Exercise

# ## Important: Kindly go through the instructions mentioned below.
# 
# - The Sheet is structured in **4 steps**:
#     1. Understanding data and manipulation
#     2. Data visualization
#     3. Implementing Machine Learning models(Note: It should be more than 1 algorithm)
#     4. Model Evaluation and concluding with the best of the model.
#     
#     
#     
# 
# - Try to break the codes in the **simplest form** and use number of code block with **proper comments** to them
# - We are providing **h** different dataset to choose from(Note: You need to select any one of the dataset from this sample sheet only)
# - The **interview calls** will be made solely based on how good you apply the **concepts**.
# - Good Luck! Happy Coding!

# ### Importing the data

# In[1]:


# use these links to do so:

import numpy as np
import pandas as pd
dataset = pd.read_csv('Churn.csv')
# ### Understanding the data

# In[2]:
#Finding out missing values and unique values present in data
#Total features present in data is 20 with total of 7043 rows
print ("\nMissing values :  ", dataset.isnull().sum().values.sum())
print ("\nUnique values :  \n",dataset.nunique())




# ### Data Manipulation

#In[3]:
#Replacing spaces with null values in total charges column
dataset['TotalCharges'] = dataset["TotalCharges"].replace(" ",np.nan)

#Dropping null values from total charges column which contain .15% missing data 
dataset = dataset[dataset["TotalCharges"].notnull()]
dataset = dataset.reset_index()[dataset.columns]

#Replacing No internet service to No
columns_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for i in columns_replace:
    dataset[i] = dataset[i].replace({'No internet service' : 'No'})
dataset["SeniorCitizen"] = dataset["SeniorCitizen"].replace({1:"Yes",0:"No"})




# In[4]:

#Separating churn and non churn customers
churn     = dataset[dataset["Churn"] == "Yes"]
not_churn = dataset[dataset["Churn"] == "No"]

# In[5]:


#Separating catagorical and numerical columns
Id_col     = ['customerID']
target_col = ["Churn"]
cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col]



# ### Data Visualization

# In[ 6]:
import warnings
warnings.filterwarnings("ignore")
import plotly.offline as py#visualization
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.figure_factory as ff#visualization
#function  for pie plot for customer attrition types
def plot_pie(column) :
    
    trace1 = go.Pie(values  = churn[column].value_counts().values.tolist(),
                    labels  = churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    domain  = dict(x = [0,.48]),
                    name    = "Churn Customers",
                    marker  = dict(line = dict(width = 2,
                                               color = "white")
                                  ),
                    hole    = .6
                   )
    trace2 = go.Pie(values  = not_churn[column].value_counts().values.tolist(),
                    labels  = not_churn[column].value_counts().keys().tolist(),
                    hoverinfo = "label+percent+name",
                    marker  = dict(line = dict(width = 2,
                                               color = "white")
                                  ),
                    domain  = dict(x = [.52,1]),
                    hole    = .6,
                    name    = "Non churn customers" 
                   )


    layout = go.Layout(dict(title = column + " distribution in customer attrition ",
                            plot_bgcolor  = "white",
                            paper_bgcolor = "white",
                            annotations = [dict(text = "churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .15, y = .5),
                                           dict(text = "Non churn customers",
                                                font = dict(size = 13),
                                                showarrow = False,
                                                x = .88,y = .5
                                               )
                                          ]
                           )
                      )
    data = [trace1,trace2]
    fig  = go.Figure(data = data,layout = layout)
    py.iplot(fig)

#for all categorical columns plot pie
for i in cat_cols :
    plot_pie(i)




# In[7]:

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#customer id col
Id_col     = ['customerID']
#Target columns
target_col = ["Churn"]
#categorical columns
cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    dataset[i] = le.fit_transform(dataset[i])
    
#Duplicating columns for multi value columns
dataset = pd.get_dummies(data = dataset,columns = multi_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(dataset[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns
df_dataset_og = dataset.copy()
dataset = dataset.drop(columns = num_cols,axis = 1)
dataset = dataset.merge(scaled,left_index=True,right_index=True,how = "left")




# In[8]:

#splitting train and test data 
from sklearn.model_selection import train_test_split
train,test = train_test_split(dataset,test_size = .2 ,random_state = 0)
##seperating dependent and independent variables
cols    = [i for i in dataset.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]






# In[1]:


### Conclusion: What all did you understand from the above charts
###From the pie graphs above I found that factors such as the tenure, monthly charges, phone service, etc.
### Tend to affect the churn rate and thus these factors must be taken into consideration while creating a model to preduct whether a customer will leave or not.
'''
The key factors affecting the churn rate were the tenure, monthly charges and contract typeand this makes sense
as customers who have stayed for longer with the compay or have a yearly contract with the company will tend to 
stay longer with the company. Also people with low monthly charge will be paying less for the services and
thus would prefer staying with the company
'''
# ### Implement Machine Learning Models

# In[9]:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import precision_score,recall_score
def telecom_churn_prediction(algorithm,training_x,testing_x,
                             training_y,testing_y,cols,cf,threshold_plot) :
    
   # baseline model
    algorithm.fit(training_x,training_y.values.ravel())
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
        
    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                              right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    print("Confusion matrix:", conf_matrix)
    
    
logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

telecom_churn_prediction(logit,train_X,test_X,train_Y,test_Y,
                         cols,"coefficients",threshold_plot = True)




# In[10]:

from imblearn.over_sampling import SMOTE

cols    = [i for i in dataset.columns if i not in Id_col+target_col]

smote_X = dataset[cols]
smote_Y = dataset[target_col]

#Split train and test data
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,
                                                                         test_size = .25 ,
                                                                         random_state = 111)

#oversampling minority class using smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=target_col)
###



logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

telecom_churn_prediction(logit_smote,os_smote_X,test_X,os_smote_Y,test_Y,
                         cols,"coefficients",threshold_plot = True)



# In[11]:

from sklearn.svm import SVC

#Support vector classifier
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)

cols = [i for i in dataset.columns if i not in Id_col + target_col]
telecom_churn_prediction(svc_lin,os_smote_X,test_X,os_smote_Y,test_Y,
                         cols,"coefficients",threshold_plot = False)






# ### Model Evaluation

# In[12]:
from sklearn.metrics import cohen_kappa_score

#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    kappa_metric = cohen_kappa_score(testing_y,predictions)
    
    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "Kappa_metric"    : [kappa_metric],
                      })
    return df
#outputs for every model
model1 = model_report(logit,train_X,test_X,train_Y,test_Y,"Logistic Regression(Baseline_model)")
model2 = model_report(logit_smote,os_smote_X,test_X,os_smote_Y,test_Y,"Logistic Regression(SMOTE)")
model3 = model_report(svc_lin,os_smote_X,test_X,os_smote_Y,test_Y,"SVM")
model_performances = pd.concat([model1,model2,model3],axis = 0).reset_index()
model_performances = model_performances.drop(columns = "index",axis =1)
table  = ff.create_table(np.round(model_performances,4))
py.iplot(table)


                      



# ### Final Conclusions

# In[ ]:
'''Based on the three models created by me the Linear Regression Baseline model had the highest Accuracy 
   and precision score followed by the SMOTE model '''
   
'''
To make the customers stay, the telecom company can offer lower monthly charges for yearly contracts 
They can also introduce loyalty benifits for people who have been with the company longer so that they 
have an incentive to stay with the company.
'''



