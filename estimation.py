def get_model(input_dim=10, output_dim=2):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dropout(0.2,))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)


def get_dataset():
    r_dielectric = 2  # m
    q2 = 5e-4  # C
    r = np.arange(1, 10)

    q1 = np.arange(start=1e-4, stop=1000e-4, step=1e-4)
    k_dielectric = np.arange(start=0.5, stop=1.5, step=.001)  # units

    X = []
    y = []
    for q1_tmp in q1:
        for k_dielectric_tmp in k_dielectric:
            F = []
            for r_tmp in r:
                F.append(force_due_to_dielectric(q1_tmp, q2, r_tmp, k_dielectric_tmp, r_dielectric))
            F = np.array(F)
            X.append(F)
            y.append([q1_tmp, k_dielectric_tmp])
    X = np.log(np.array(X))
    y = np.array(y)
    return X, y


X, y = get_dataset()
X_train, X_test, y_train, y_test, y_old_train, y_old_test = train_test_split(X, scaler.fit_transform(y), y, test_size=0.33, random_state=42)
model = get_model(X_train.shape[-1], y_train.shape[-1])
model.fit(X_train, y_train, epochs=5)
y_pred = model.predict(X_test)
print(y_old_test[0], scaler.inverse_transform(y_pred[0]))
print(y_old_test[54], scaler.inverse_transform(y_pred[54]))
