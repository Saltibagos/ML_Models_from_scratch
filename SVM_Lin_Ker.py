import numpy as np





#SVM with linear kernel
class SVM:
    def __init__(self,learning_rate,n_iters=100,lambda_param=0.01):          #The constructor of an SVM object requires 3 hyperparameters ie learning_rate for Gradient descentn number of
        self.learning_rate=learning_rate                                     # iterations and a regularization parameter for the soft margin
        self.n_iters=n_iters
        self.lambda_param=lambda_param




    def fit(self,X,y):
        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.y=y
        for i in range(self.n_iters):            #The weights and bias are updated via Gradient Descent over n_iter iterations
            self.gradient_descent_update()




    def gradient_descent_update(self):              #Gradient descent method that minimizes the hinge loss function
        y_label=np.where(self.y<=0,-1,1)
        for i, x_i in enumerate(self.X):
            c=y_label[i]*(np.dot(x_i,self.w)-self.b)>=1
            if c:
                dw=2*self.lambda_param*self.w
                db=0
            else:
                dw=2*self.lambda_param*self.w-np.dot(x_i,y_label[i])
                db=y_label[i]
            self.w=self.w-self.learning_rate*dw
            self.b=self.b-self.learning_rate*db



    def predict(self,X):
        output=np.dot(X,self.w)-self.b
        p_labels=np.sign(output)
        return np.where(p_labels<=-1,0,1)


