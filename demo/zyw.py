#导入类库和加载数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_names = ["date",
               "price",
               "bedroom_num",
               "bathroom_num",
               "house_area",
               "park_space",
               "floor_num",
               "house_score",
               "covered_area",
               "basement_area",
               "yearbuilt",
               "yearremodadd",
               "lat",
               "long"]
data = pd.read_csv("kc_train.csv",names=train_names)
data.head()
# 拆分数据：把原始数据中的年月日拆开，然后根据房屋的建造年份和修复年份计算一下售出时已经过了多少年，这样就有17个特征。
sell_year,sell_month,sell_day=[],[],[]
house_old,fix_old=[],[]
for [date,yearbuilt,yearremodadd] in data[['date','yearbuilt','yearremodadd']].values:
    year,month,day=date//10000,date%10000//100,date%100
    sell_year.append(year)
    sell_month.append(month)
    sell_day.append(day)
    house_old.append(year-yearbuilt)
    if yearremodadd==0:
        fix_old.append(0)
    else:
        fix_old.append(year-yearremodadd)
del data['date']
data['sell_year']=pd.DataFrame({'sell_year':sell_year})
data['sell_month']=pd.DataFrame({'sell_month':sell_month})
data['sell_day']=pd.DataFrame({'sell_day':sell_day})
data['house_old']=pd.DataFrame({'house_old':house_old})
data['fix_old']=pd.DataFrame({'fix_old':fix_old})
data.head()

#特征缩放
data = data.astype('float')
x = data.drop('price',axis=1)
y = data['price']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
newX= scaler.fit_transform(x)
newX = pd.DataFrame(newX, columns=x.columns)
newX.head()
# 划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(newX, y, test_size=0.2, random_state=21)

#模型建立 决策树
from sklearn import metrics
def RF(X_train, X_test, y_train, y_test):    #随机森林
    from sklearn.ensemble import RandomForestRegressor
    model= RandomForestRegressor(n_estimators=200,max_features=None)
    model.fit(X_train, y_train)
    predicted= model.predict(X_test)
    mse = metrics.mean_squared_error(y_test,predicted)
    return (mse/10000)
# 模型建立 逻辑回归
def LR(X_train, X_test, y_train, y_test):    #线性回归
    from sklearn.linear_model import LinearRegression
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    predicted = LR.predict(X_test)
    mse = metrics.mean_squared_error(y_test,predicted)
    return (mse/10000)

print('RF mse: ',RF(X_train, X_test, y_train, y_test))
print('LR mse: ',LR(X_train, X_test, y_train, y_test))

