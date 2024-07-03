import numpy as np

class LogisticRegression():
    def __init__(self,learing_rate,epochs,threshold=0.5):
        self.learning_rate=learing_rate
        self.epochs=epochs
        self.threshold=threshold
    
    @staticmethod
    def sigmoid(x):
        s=1/(1+np.exp(-x))
        return s
       
    def fit(self,train_input,train_output):
        self.m , self.n = train_input.shape
        self.X=train_input
        self.Y=train_output
        self.W=np.zeros(self.n)
        self.B=0

        for i in range(self.epochs):
            self.back_propogation()

        return self

    def forward_propogation(self):
        self.z=np.multiply(self.X,self.W)+self.B
        probabilty=self.sigmoid(self.z)
        return probabilty
    
    
    def predict(self,test_input):
        z=np.dot(test_input,self.W)+self.B
        probabilty=self.sigmoid(z)

        m,n=test_input.shape
        predict=np.zeros(m)

        for i in range(m):
            if(probabilty[i][0]>=0.5):
                predict[i]=1
            else:
                predict[i]=0
        
        return predict,probabilty



 
    def back_propogation(self):
        prediction=self.forward_propogation()
        difference=self.Y-prediction
        dw=-2*(self.X.T).dot(difference)/self.m
        db=-2*(np.sum(difference))/self.m

        self.W=self.W - self.learning_rate * dw
        self.B=self.B - self.learning_rate * db

        return self






