import numpy as np 
import h5py
import matplotlib.pyplot as plt 
from  IPython import get_ipython

#get_ipython().run_line_magic('matplotlib','inline')
#% matplotlib inline

plt.rcParams['figure.figsize']= (5.0, 4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

np.random.seed(1)

def zero_pad(X,pad):
    
    return np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)) ,'constant')


def zero_pad_test():
    np.random.seed(1)
    x= np.random.randn(4,3,3,2)
    x_pad = zero_pad(x,2)
    print x_pad

    print ("x.shape=",x.shape)
    print("x.pad.shape=",x_pad.shape)

    fig,axarr= plt.subplots(1,2)

    axarr[0].set_title('x')
    axarr[0].imshow(x[0,:,:,0])

    axarr[1].set_title('x_pad')
    axarr[1].imshow(x_pad[0,:,:,0])

    plt.show()


#zero_pad_test()

def conv_single_step(a_slice_prev,W,b):
    s= np.multiply(a_slice_prev,W)

    Z = np.sum(s)

    return float(b)+Z


def conv_single_step_test():
    np.random.seed(1)
    a_slice_prev = np.random.randn(4,4,3)
    W = np.random.randn(4,4,3)
    b = np.random.randn(1,1,1)

    Z =conv_single_step(a_slice_prev,W,b)
    print("Z=",Z)
    return Z

conv_single_step_test()


def conv_forward(A_prev,W,b,hparameters):
    (m,n_H_prev,n_C_prev,n_C) = A_prev.shape

    (f,f,n_C_prev,n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - f+2*pad) /stride) +1
    n_W = int((n_W_prev -f+2*pad)/stride) +1

    Z= np.zeros((m,n_H,n_W,n_C))

    A_prev_pad =zero_pad(A_prev,pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h *stride
                    vert_end = vert_start + f  

                    horz_start = w*stride
                    horz_end = horz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horz_start:horz_end,:]

                    #Convolve the 3D slice
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[...,c],b[...,c])

                    
    assert(Z.shape==(m,n_H,n_W,n_C))
    cache = (A_prev, W,b,hparameters)

    return Z,cache


def pool_forward(A_prev,hparameters, mode='max'):
    # get shape of the input 
    (m,n_H,n_W,n_C) = A_prev.shape

    #Get filter size and stride for pool layer

    pad = hparameters['pad']
    stride = hparameters['stride']
    f= hparameters['f']

    # create output layer . Initialize to zeros
    A = np.zero((m,n_H,n_W,n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_begin = h* stride
                    v_end = v_begin  + f

                    h_begin = w* stride
                    h_end = h_begin + f

                    a_slice = A_prev[i,v_begin:v_end, h_begin:h_end, c]

                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_slice)
                    elif mode =='average':
                        A[i,h,w,c] = np.mean(a_slice)

    cache = (A_prev, hparameters) 

    return A,cache               
