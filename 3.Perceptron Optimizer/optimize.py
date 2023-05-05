import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

x,y = make_classification(
    n_samples=100, # 100 observations 
    n_features=2, # 5 total features
    n_informative=1,
    n_redundant=0, # 3 'useful' features
    n_classes=2, # binary target/label 
    random_state=41,
    n_clusters_per_class=1,
    hypercube=False,
    class_sep=15 # if you want the same results as mine
)

# plt.figure(figsize=(10,6))
# plt.scatter(x[:,0],x[:,1],c=y)
# plt.show()

y[y==0] = -1

class Perceptron:
    def __init__(self,lr):
        self.w1 = 1
        self.w2 = 1
        self.b = 1
        self.lr = lr

    def fit(self,x,y):
        for p in range(100):
            for i in range(x.shape[0]):
                z = self.w1*x[i][0] + self.w2*x[i][1] + self.b
                print()
                print(f"z = {z}--")
                print(f"y[i]*z={-y[i]*z}--   ")
                print(f" y = {y[i]}")
                print(f"i = {i}")

                if (-y[i]*z) > 0:
                    self.w1 = self.w1 + self.lr*(-y[i]*x[i][0])
                    self.w2 = self.w2 + self.lr*(-y[i]*x[i][1])
                    self.b = self.b + self.lr*(-y[i])
                    print(f"{i} ...w1 = {self.w1} , w2 = {self.w2} , b = {self.b}")


            
            # m = -(self.w1)/self.w2
            # bo = -(self.b)/self.w2
            # x_input = np.linspace(-3,3,100)
            # y_input = m*x_input + bo

            # plt.figure(figsize=(10,6))
            # plt.plot(x_input,y_input,color='red')
            # plt.scatter(x[:,0],x[:,1],c=y)
            # plt.show()
        return self.w1,self.w2,self.b
    
    def show(self):
        w1,w2,bo = self.fit(x,y)
        print(f"w1 = {w1} , w2 = {w2} , b = {bo}")
        m = -(w1)/w2
        b = -(bo)/w2
        x_input = np.linspace(-3,3,100)
        y_input = m*x_input + b

        plt.figure(figsize=(10,6))
        plt.plot(x_input,y_input,color='red')
        plt.scatter(x[:,0],x[:,1],c=y)
        plt.show()

obj = Perceptron(0.05)
obj.show()