import numpy as np, json
from pathlib import Path
p=Path(__file__).resolve().parent
np.set_printoptions(precision=12,suppress=True,linewidth=160)
# Store matrices in Custom_MLP convention (neurons, inputs). Compute analytic reference using transposed views.
W0=np.array([[.10,-.05,.08,.04],[.06,.10,-.03,.07],[-.04,.02,.09,.05],[.07,-.02,.01,.08],[.03,.06,.04,-.01]],dtype=np.float64)
b0=np.array([.20,.15,.18,.12],dtype=np.float64)
W1=(np.arange(64,dtype=np.float64).reshape(4,16)%9-4)*.012
b1=np.full(16,.25,dtype=np.float64)
W2=np.array([[.06,-.04] if i%2==0 else [-.03,.05] for i in range(16)],dtype=np.float64)
b2=np.array([.02,-.01],dtype=np.float64)
x=np.array([[1.,.5,-.5,2.,1.5]],dtype=np.float64)
y=np.array([[1.,0.]],dtype=np.float64)
z0=x@W0+b0; a0=np.maximum(z0,0)
z1=a0@W1+b1; a1=np.maximum(z1,0)
z2=a1@W2+b2
s=np.exp(z2-z2.max(axis=1,keepdims=True)); prob=s/s.sum(axis=1,keepdims=True)
loss=-np.sum(y*np.log(prob))
dz2=prob-y
dW2=a1.T@dz2; db2=dz2.sum(axis=0)
da1=dz2@W2.T; dz1=da1*(z1>0)
dW1=a0.T@dz1; db1=dz1.sum(axis=0)
da0=dz1@W1.T; dz0=da0*(z0>0)
dW0=x.T@dz0; db0=dz0.sum(axis=0)
assert np.all(z0>0) and np.all(z1>0)
layers=[{'layertype':'Dense','neurons':4},{'layertype':'Activation','type':'ReLu'},{'layertype':'Dense','neurons':16},{'layertype':'Activation','type':'ReLu'},{'layertype':'Dense','neurons':2},{'layertype':'Activation','type':'Softmax'}]
np.savez_compressed(p/'weights.npz',formatversion=1,inputs=5,layers=json.dumps(layers),weights_0=W0.T,biases_0=b0.reshape(-1,1),weights_1=W1.T,biases_1=b1.reshape(-1,1),weights_2=W2.T,biases_2=b2.reshape(-1,1))
np.savez_compressed(p/'trainingsdaten.npz',x_data=x,y_data=y)
np.savez_compressed(p/'expected_gradients.npz',dW_0=dW0.T,db_0=db0.reshape(-1,1),dW_1=dW1.T,db_1=db1.reshape(-1,1),dW_2=dW2.T,db_2=db2.reshape(-1,1),softmax_output=prob,loss=loss,z_0=z0,z_1=z1,z_2=z2)
print('loss',loss,'softmax',prob)
for i,(dw,db) in enumerate([(dW0,db0),(dW1,db1),(dW2,db2)]): print(i,'dW',dw.shape,'db',db.shape,'db values',db,'dW first row',dw[0])
print('files: ',[q.name for q in p.glob('*.npz')])
