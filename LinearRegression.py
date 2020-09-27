import xlrd
from pylab import *
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

data = xlrd.open_workbook(r'C:\Users\ChenJL\Desktop\\data5.xlsx')
table = data.sheets()[0]
cols = table.ncols
tables = []
def import_excel(excel):
    for col in range(cols):
        print(cols)
        array=excel.col_values(col,start_rowx=0,end_rowx=None)
        tables.append(array)
    return tables
import_excel(table)


x_f = np.array(tables[1])
y_f = np.array(tables[2])
x = x_f.reshape(1171,1)
y = y_f.reshape(1171,1)
x_train = x[:-500]
x_test = x[-500:]
y_train = y[:-500]
y_test = y[-500:]

lr = linear_model.LinearRegression()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)

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


figure(figsize=(13,16), dpi=80)
plt.scatter(x_test, y_test,  color='black',label = 'data point')
plt.plot(x_test, y_pred, color='red', linewidth=1,label = 'Curve fitting')
legend(loc='upper left')
X = x_test
Y = y_test
xmin ,xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()

dx = (xmax - xmin) * 0.05
dy = (ymax - ymin) * 0.05

xlim(xmin - dx, xmax + 1.5*dx)
ylim(ymin - dy, ymax + 1.5*dy)
plt.xticks(np.linspace(xmin - dx,xmax + dx,20,endpoint=True))
plt.yticks(np.linspace(ymin - dy,ymax + dy,10,endpoint=True))
annotate(r'y = 0.0124x+42.5147',
         xy=(107.78, 43.851172), xycoords='data',
         xytext=(0.4, 0.65), textcoords='axes fraction', fontsize=15,
         arrowprops = dict(facecolor='red',arrowstyle="->", connectionstyle="arc3,rad=.2"))


savefig('C:\\Users\ChenJL\pictures\\up1.png',dpi=72)
plt.show()