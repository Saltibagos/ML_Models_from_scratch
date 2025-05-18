import numpy as np

sigmoid=lambda x: 1/(1+np.exp(-x))

#A simple Logistic regression classifier
class LogisticRegression:
    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.learning_rate=learning_rate
        self.n_iters=n_iters



    def fit(self,X,y):
        self.m, self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.y=y
        for _ in range(self.n_iters):
            self.gradient_descent_update()


    def gradient_descent_update(self):
        y_bar=sigmoid(self.X.dot(self.w)+self.b)
        dw=(1/self.m)*np.dot(self.X.T,(y_bar-self.y))
        db=(1/self.m)*np.sum(y_bar-self.y)
        self.w=self.w-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db




    def predict(self,X):
        sig=sigmoid(X.dot(self.w)+self.b)
        prediction=np.where(sig>0.5,1,0)
        return prediction