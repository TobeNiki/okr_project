import numpy as np


class LinearRegression:

    def __init__(self, optimization:str='least'):
        self.optimization = optimization

    def __input_set(self, x, y):
        self.x = x
        self.y = y
        self.N = x.shape[0]
        self.xDim = x.shape[1]

    def fit(self, x, y):
        self.__input_set(x, y)
        if self.optimization == 'least':
            self.___LeastSquares()
        elif self.optimization == 'l2norm':
            self.___L2NormRegularizationLeastSquares()
        else:
            raise ValueError('Noting Optimization Names!')

    def ___LeastSquares(self):
        z = np.concatenate([self.x, np.ones([self.N,1])], axis=1)
        #分母の計算
        zz = 1/self.N * np.matmul(z.T, z)
        #分子の計算
        zy = 1/self.N * np.matmul(z.T, self.y)
        #パラメータの最適化
        v = np.matmul(np.linalg.inv(zz),zy)
        self.w = v[:-1]
        self.b = v[-1]

    def ___L2NormRegularizationLeastSquares(self):
        pass

    def predict(self, x):
        return np.matmul(x, self.w) + self.b
    
    def RMES(self, x, y)->float:
        return np.sqrt(np.mean(np.square(self.predict(x)-y)))

    def R2(self, x, y)->float:
        numerator = np.sum(np.square(self.predict(x)-y))
        denominator = np.sum(np.square(y-np.mean(y,axis=0)))
        return 1 - denominator/numerator

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df['MEDV'] = boston.target

    X = boston_df[['RM']].values
    Y = boston_df[['MEDV']].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 0) # データを学習用と検証用に分割
    
    lr = LinearRegression()
    lr.fit(X_train,Y_train)
    Y_pred = lr.predict(X_train)
    print(lr.w,lr.b)
    print(lr.R2(Y_train, Y_pred))
    plt.scatter(X, Y, color = 'blue')         # 説明変数と目的変数のデータ点の散布図をプロット
    plt.plot(X, lr.predict(X), color = 'red') # 回帰直線をプロット

    plt.title('Regression Line')               # 図のタイトル
    plt.xlabel('Average number of rooms [RM]') # x軸のラベル
    plt.ylabel('Prices in $1000\'s [MEDV]')    # y軸のラベル
    plt.grid()                                 # グリッド線を表示
    plt.show()