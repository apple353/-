import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import pywt
import statsmodels.api as sm
from  statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.ar_model import AR
import math
data = pd.read_excel(r'D:\data_zgz\data.xlsx',sheet_name='qxwy1')
test_weiyi = np.array(data['value'])[:-30]
print(len(test_weiyi))
test_date = np.array(data['date'])[:-30]
pred_weiyi = np.array(data['value'])[-30:]
pred_date = np.array(data['date'])[-30:]
A2,D2,D1 = pywt.wavedec(test_weiyi,'db4',mode='sym',level=2)#D.为每一层分解得到的高频信号;A为第n层的低频信号
coeff = [A2,D2,D1]
n=[len(D1),len(D2)]
print(n)

a=0
for i in range(len(D1)):
    a += D1[i]-np.mean(D1)
sigma = a/len(D1)
na = []
def namuda(n):
    for i in  range(len(n)):
        x = sigma*(np.log(n[i]))**0.5
        na.append(x)
    return na
na = namuda(n)
print(na)


#得到ＡＲＭＡ模型系数
arma_qxwy1 = sm.tsa.arma_order_select_ic(test_weiyi,ic = 'aic')['aic_min_order']#根据aic准测选取系数
'''arma_D2 = sm.tsa.arma_order_select_ic(D2,ic = 'aic')['aic_min_order']
arma_D1 = sm.tsa.arma_order_select_ic(D1,ic = 'aic')['aic_min_order']
print(arma_A2,arma_D2,arma_D1)'''
#ARMA模型
model_qxwy1 = ARMA(test_weiyi,order=arma_qxwy1)
'''model_D2 = ARMA(D2,order=arma_D2)
model_D1 = ARMA(D1,order=arma_D1)'''
result_qxwy1 = model_qxwy1.fit()
'''result_D2 = model_D2.fit()
result_D1 = model_D1.fit()'''

'''plt.subplot(312)
plt.plot(D2,'red')
plt.plot(result_D2.fittedvalues,'blue')
plt.title('D2')
plt.subplot(313)
plt.plot(D1,'red')
plt.plot(result_D1.fittedvalues,'blue')
plt.title('D1')
plt.show()
#分解所有的序列
A2_all,D2_all,D1_all = pywt.wavedec(np.array(data['weiyi']),'db4',mode='sym',level=2)
detal = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)]
print(detal)'''
pqxwy1 = model_qxwy1.predict(params=result_qxwy1.params,start=0,end=len(data['value'])+10)
plt.figure()
plt.plot(data['value'],'red')
plt.plot(result_qxwy1.fittedvalues,'blue')
plt.plot(pqxwy1,'green')
plt.title('qxwy')
plt.show()
'''pD2 = model_D2.predict(params=result_D2.params,start=0,end=len(D2)+detal[1])
pD1 = model_D1.predict(params=result_D1.params,start=0,end=len(D1)+detal[1])'''

'''for i in range(1,len(coeff)):
    coeff[i]=pywt.threshold(coeff[i],value=na[i-1],mode='hard')

#重构
rsc_data = pywt.waverec(coeff,'db4',mode='sym')
plt.figure
plt.plot(data['value'],'blue')
plt.plot(rsc_data,'red')
plt.show()'''























