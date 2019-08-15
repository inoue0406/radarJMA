#
# Scalers for rainfall dataset
# 

import numpy as np

class LogScaler():
    # class for scaling of rainfall values
    def __init__(self):
        # coeff for intensity -> reflecivity
        self.a = 256.0
        self.b = 1.42
        self.rmax = 201.0 # max rainfall intensity
        self.c1 = 0.0
        self.c2 = self.rfl2dbz(self.mmh2rfl(self.rmax))
        
    # conversion between mm/h <-> reflectivity
    def mmh2rfl(self, r, a=256., b=1.42):
        return a * r ** b
    def rfl2mmh(self, z, a=256., b=1.42):
        return (z / a) ** (1. / b)
    # conversion between reflectivity <-> magnitude
    def rfl2dbz(self, z):
        return 10. * np.log10(z)
    def dbz2rfl(self, d):
        return 10. ** (d / 10.)

    def fwd(self,X):
        # forward transformation
        # input X: numpy array
        #          rainfall intensity with 0-201[mm/h] value range
        X_rfl = self.mmh2rfl(X) # intensity to reflectivity
        X_rfl[X_rfl < 0.1] = 0.1
        X_dbz = self.rfl2dbz(X_rfl) # reflectivity to dBz
        Xscl = ((X_dbz-self.c1)/(self.c2-self.c1)) # output scaled to [0-1]
        return Xscl

    def inv(self,X_scl):
        # inverse transformation
        # input X_scl: numpy array
        #          scaled intensity with 0-1 value range
        X_rfl = self.dbz2rfl((X_scl)*(self.c2 - self.c1) + self.c1)
        X_mmh = self.rfl2mmh(X_rfl)
        # interpret "small enough" value as zero
        X_mmh[X_rfl <= 0.1] = 0.0
        return X_mmh

class LinearScaler():
    # class for linear scaling of rainfall values
    def __init__(self,rmax=201.0):
        self.rmax = 201.0 # max rainfall intensity

    def fwd(self,X):
        # forward transformation
        # input X: numpy array
        #          rainfall intensity with 0-201[mm/h] value range
        return X/self.rmax

    def inv(self,X_scl):
        # inverse transformation
        # input X_scl: numpy array
        #          scaled intensity with 0-1 value range
        return X_scl*self.rmax
    
class RootScaler():
    # class for root scaling of rainfall values
    def __init__(self):
        self.rmax = 201.0 # max rainfall intensity
        self.a = 1/3.0    # "cubic root"

    def fwd(self,X):
        # forward transformation
        # input X: numpy array
        #          rainfall intensity with 0-201[mm/h] value range
        X = np.maximum(X,0.0) # force the value to be positive
        return (X/self.rmax)**self.a

    def inv(self,X_scl):
        # inverse transformation
        # input X_scl: numpy array
        #          scaled intensity with 0-1 value range
        X_scl = np.maximum(X_scl,0.0) # force the value to be positive
        return self.rmax*(X_scl**(1/self.a))
    
class RootIntScaler():
    # input: integer variable
    # output: inverse scaled data (float)
    def __init__(self):
        self.rmax = 201.0 # max rainfall intensity
        self.a = 1/3.0    # "cubic root"

    def fwd(self,X):
        # forward transformation
        # input X: numpy array
        #          rainfall intensity with 0-255 integer value range
        return X/255.0

    def inv(self,X_scl):
        # inverse transformation
        # input X_scl: numpy array
        #          scaled intensity with 0-1 value range
        X_scl = np.maximum(X_scl,0.0) # force the value to be positive
        return self.rmax*(X_scl**(1/self.a))
    
if __name__ == '__main__':
    # test for scaling class
    print('Log Scaler----------------------')
    scl = LogScaler()
    print('test for ordinary values')
    X = np.array([0.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.001])
    print('*note that small values are transformed to zero*')
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.01])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.1])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([1.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([10.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([201.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
#    for n in range(9):
#        X = np.array([(n*0.001+0.001)])
#        print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
#    for n in range(9):
#        X = np.array([(n*0.01+0.01)])
#        print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
#    for n in range(9):
#        X = np.array([(n*0.1+0.1)])
#        print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
#    for n in range(201):
#        X = np.array([(n*1.0+1.0)])
#        print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
        
    print('Root Scaler----------------------')
    scl = RootScaler()
    print('test for ordinary values')
    X = np.array([0.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.001])
    print('*note that small values are transformed to zero*')
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.01])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.1])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([1.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([10.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([201.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    
    print('test for irregular values')
    X = np.array([-1.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([250.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    
    print('Linear Scaler----------------------')
    scl = LinearScaler()
    print('test for ordinary values')
    X = np.array([0.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.001])
    print('*note that small values are transformed to zero*')
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.01])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([0.1])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([1.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([10.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([201.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    
    print('test for irregular values')
    X = np.array([-1.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))
    X = np.array([250.0])
    print('X=%f,X_scl=%f,X_inv=%f' % (X,scl.fwd(X),scl.inv(scl.fwd(X))))


