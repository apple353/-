# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from pylab import *
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# %%
data = pd.read_excel('data5.xlsx')
height,width = df.shape
x = np.zeros((height,width))
for i in xrange(0,height):
	for j in xrange(1,width+1): 
		x[i][j-1] = df.ix[i,j]


# %%
tables = np.zeros((height,width))
for i in xrange(0,height):
    for j in xrange(1,width+1): 
        x[i][j-1] = df.ix[i,j]


# %%
x_i = 


# %%
a = np.zeros(shape=(500,1))
k = np.zeros(shape=(1,1))
b = np.zeros(shape=(1,1))


# %%
for i in range(0,cols-1,2):   
    x_i = (tables[i])
    y_i =(tables[i+1])
    xi = x_i.reshape(rows,1)
    yi = y_i.reshape(rows,1)
    x_train = xi[:-500]
    x_test = xi[-500:]
    a = np.append(a,x_test,axis=1)
    y_train = yi[:-500]
    y_test = yi[-500:]
    a = np.append(a,y_test,axis=1)
    lr = linear_model.LinearRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    a = np.append(a,y_pred,axis=1)
    k = np.append(k,lr.coef_,axis=1)
    b = np.append(b,[lr.intercept_],axis=1)
    print('Coefficients:\n',lr.coef_)
    print('b:',lr.intercept_)
    print('Mean squared error:%.2f'% mean_squared_error(y_test, y_pred))
    print('Coefficient of determination:%.2f'%r2_score(y_test,y_pred))
    if abs(r2_score(y_test,y_pred))<0.3:
        print('低度线性相关.')
    elif 0.3<abs(r2_score(y_test,y_pred))<0.7:
         print('中度相关.')
    else:
        print('高度相关.')
   


# %%
data = np.delete(a,0,axis=1)
list_k = np.delete(k,0,axis=1)
list_b = np.delete(b,0,axis=1)


# %%
figure(figsize=(15,18), dpi=80)
for i in range(0,points):
    subplot(3,3,i+1) 
    plt.scatter(data[:,3*i], data[:,3*i+1],color = 'blue',label = 'up{} data point'.format(i+1))
    plt.plot(data[:,3*i],data[:,3*i+2], linewidth=1,color = 'red',label = 'up{} Curve fitting'.format(i+1))
    X = data[:,3*i]
    Y = data[:,3*i+1]
    xmin ,xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()

    dx = (xmax - xmin) * 0.05
    dy = (ymax - ymin) * 0.05

    xlim(xmin - dx, xmax + 1.5*dx)
    ylim(ymin - dy, ymax + 1.5*dy)
    plt.xticks(np.linspace(xmin - dx,xmax + dx,10,endpoint=True))
    plt.yticks(np.linspace(ymin - dy,ymax + dy,10,endpoint=True))
    annotate(r'UP{}:y = {:.4f}x+{:.2f}'.format(i+1,list_k[0,i],list_b[0,i]),xy=(107.78, 43.851172), 
             xycoords='data', xytext=(0.3, 0.65), textcoords='axes fraction', fontsize=12)
savefig('C:\\Users\ChenJL\pictures\\up1.png',dpi=72)        


# %%



# %%



