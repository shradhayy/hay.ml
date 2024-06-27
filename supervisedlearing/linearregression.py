import numpy as np
class linearregression():

    def __init__(self,learning_rate,iterations):
        self.learning_rate=learning_rate
        self.iterations=iterations

    def fit(self,train_input,train_output):
        self.m , self.n = train_input.shape
        self.train_input=train_input
        self.train_output=train_output
        self.wieghts=np.zeros(self.n)
        self.bais=0

        for i in range(self.iterations):
            self.back_propogation()

        return self

    def forward_propogation(self,train_input):
        return train_input.dot(self.wieghts)+self.bais
    

    def back_propogation(self):
        predictions=self.forward_propogation(self.train_input)
        difference=self.train_output-predictions
        dw=-2*(self.train_input.T).dot(difference)/self.m
        db=-2*(np.sum(difference))/self.m

        self.wieghts=self.wieghts-self.learning_rate*dw
        self.bais=self.bais-self.learning_rate*db

        return self
    







    


    
    


    
        
    
    
