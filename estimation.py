import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from sklearn import preprocessing

epsilon_naught = 8.8541878e-12 #Fm^-1


def coulomb_law(q1,q2,r,epsilon_r=1):
    return q1*q2/(4*math.pi*epsilon_naught*epsilon_r*r*r)


def force_due_to_dielectric(q1,q2,r,k_dielectric,r_dielectric):
    epsilon_r_dielectric=1/k_dielectric 
    if r < r_dielectric:
        return coulomb_law(q1,q2,r,epsilon_r_dielectric)
    else:
        return coulomb_law(q1,q2,r,1)

def get_model(input_dim=10,output_dim=2):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dropout(0.2,))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

scaler = preprocessing.StandardScaler()

def gen_sample(q1,k_dielectric):
    for q1_tmp in q1:
        for k_dielectric_tmp in k_dielectric:
            F=[]
            for r_tmp in r:
                F.append(force_due_to_dielectric(q1_tmp,q2,r_tmp,k_dielectric_tmp,r_dielectric))
            F=np.array(F)
            X.append(F)
            y.append([q1_tmp,k_dielectric_tmp])
    X=np.log(np.array(X))
    y=np.array(y)
    y=scaler.transform(y, norm='l1')
    return X,y

def get_dataset():
    r_dielectric = 2#m
    q2 = 5e-4#C
    r= np.arange(1,10)
    
    q1 = np.arange(start=1e-4,stop=1000e-4,step=1e-4)
    k_dielectric = np.arange(start=0.5,stop=1.5,step=.001) #units

    q1_avg= np.mean(q1)
    k_dielectric_avg = np.mean(k_dielectric)
    
    X = []
    y=[]
    for q1_tmp in q1:
        for k_dielectric_tmp in k_dielectric:
            F=[]
            for r_tmp in r:
                F.append(force_due_to_dielectric(q1_tmp,q2,r_tmp,k_dielectric_tmp,r_dielectric))
            F=np.array(F)
            X.append(F)
            y.append([q1_tmp,k_dielectric_tmp])
    X=np.log(np.array(X))
    y=np.array(y)
    y=scaler.fit_transform(y)
    return train_test_split( X, y, test_size=0.33, random_state=42)

if __name__=="__main__":
    X_train, X_test, y_train, y_test, = get_dataset()
    model = get_model(X_train.shape[-1],y_train.shape[-1])
    model.fit(X_train,y_train,epochs=5)
    c=model.predict(X_test)

