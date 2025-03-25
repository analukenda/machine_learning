import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

n_data_points = 500
np.random.seed(69)

# Generiraj podatke o bodovima na prijamnom ispitu koristeći normalnu razdiobu i ograniči ih na interval [1, 3000].
exam_score = np.random.normal(loc=1500.0, scale = 500.0, size = n_data_points)
exam_score = np.round(exam_score)
exam_score[exam_score > 3000] = 3000
exam_score[exam_score < 0] = 0

# Generiraj podatke o ocjenama iz srednje škole koristeći normalnu razdiobu i ograniči ih na interval [1, 5].
grade_in_highschool = np.random.normal(loc=3, scale = 2.0, size = n_data_points)
grade_in_highschool[grade_in_highschool > 5] = 5
grade_in_highschool[grade_in_highschool < 1] = 1

# Matrica dizajna.
grades_X = np.array([exam_score,grade_in_highschool]).T

# Završno, generiraj izlazne vrijednosti.
rand_noise = np.random.normal(loc=0.0, scale = 0.5, size = n_data_points)
exam_influence = 0.9
grades_y = ((exam_score / 3000.0) * (exam_influence) + (grade_in_highschool / 5.0) \
            * (1.0 - exam_influence)) * 5.0 + rand_noise
grades_y[grades_y < 1] = 1
grades_y[grades_y > 5] = 5

plt.subplot(1,2,1)
plt.scatter(exam_score,grades_y,color='blue')
plt.xlabel('Rezultat ispita')
plt.ylabel('Konačna ocjena')
plt.subplot(1,2,2)
plt.scatter(grade_in_highschool,grades_y,color='blue')
plt.xlabel('Ocjene u školi')
plt.ylabel('Konačna ocjena')
plt.show()

model=Ridge(alpha=0.01)
model.fit(grades_X,grades_y)
pred=model.predict(grades_X)
err=mean_squared_error(grades_y,pred)
print('Greška modela ',err)
print(model.coef_)

scaler=StandardScaler()
scaler.fit(grades_X)
grades_X_fixed=scaler.transform(grades_X)
scaler.fit(np.reshape(grades_y,(n_data_points,1)))
grades_y_fixed=scaler.transform(np.reshape(grades_y,(n_data_points,1)))
model=Ridge(alpha=0.01)
model.fit(grades_X_fixed,grades_y_fixed)
pred=model.predict(grades_X_fixed)
err=mean_squared_error(grades_y_fixed,pred)
print('Greška modela ',err)
print(model.coef_[0])

grades_X_fixed_colinear=[]
for i in grades_X_fixed:
    arr=list(i)
    arr.append(arr[-1]*2)
    grades_X_fixed_colinear.append(arr)
grades_X_fixed_colinear=np.array(grades_X_fixed_colinear)

model=Ridge(alpha=0.01)
model.fit(grades_X_fixed_colinear,grades_y_fixed)
pred=model.predict(grades_X_fixed_colinear)
err=mean_squared_error(grades_y_fixed,pred)
print('Greška modela ',err)
print(model.coef_[0])

w_1=[]
w_2=[]
indexi=range(500)
for i in range(10):
    rand_indexi=np.random.choice(indexi,250,replace=False)
    x=[]
    y=[]
    for j in rand_indexi:
        x.append(grades_X_fixed_colinear[j])
        y.append(grades_y_fixed[j])
    model = Ridge(alpha=0.01)
    model.fit(x, y)
    coef_1 = model.coef_[0]
    coef_1[0] = model.intercept_
    w_1.append(coef_1)

    model = Ridge(alpha=100)
    model.fit(x, y)
    coef_2 = model.coef_[0]
    coef_2[0] = model.intercept_
    w_2.append(coef_2)
    print('Test ' + str(i + 1) + ' => lambda = 0.01:' + str(coef_1) + ', lambda = 100:' + str(coef_2))
w_1=np.array(w_1)
w_2=np.array(w_2)
for i in range(3):
    w_i1=w_1[:,i]
    w_i2=w_2[:,i]
    avg_1=np.mean(w_i1)
    avg_2=np.mean(w_i2)
    sum1=0
    sum2=0
    for j in range(10):
        sum1+=((w_i1[j]-avg_1)**2)
        sum2 += ((w_i2[j] - avg_2) ** 2)
    dev1=(sum1/10)**0.5
    dev2=(sum2/10)**0.5
    print('Devijacija težine w'+str(i)+' za faktor 0.01: '+str(dev1))
    print('Devijacija težine w' + str(i) + ' za faktor 100: ' + str(dev2))