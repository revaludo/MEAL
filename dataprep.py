#coding=utf-8
import numpy as np
import wmf

def load_matrix(filename,alpha):
    R=np.loadtxt(filename)
    num_users=R.shape[0]
    num_items=R.shape[1]
    M= R.copy()
    for user in range(num_users):
        for item in range(num_items):
            count = R[user][item]
            if count != 0:
                M[user, item] = 1
    S = M.copy()
    S.data = 1 + alpha * S.data
    return S,M,num_users,num_items


def genernatebatch(img_file,label_file,matrix_file,batch_num):
    S,M,um,vm = load_matrix(matrix_file.txt, alpha=15)
    U, V = wmf.factorize(S,M, num_factors=50, lambda_reg=1e-3, num_iterations=200, init_std=0.01,
                         dtype='float32')
    ubsize=um/batch_num
    vbsize=vm/batch_num

    up=np.random.permutation(np.arange(um))
    vp = np.random.permutation(np.arange(vm))

    imgs = np.load(img_file)
    labels = np.load(label_file)

    for batch in range(batch_num):
        if batch <batch_num-1:
            batchu=up[batch*ubsize,(batch+1)*ubsize]
            batchv=vp[batch * vbsize, (batch + 1) * vbsize]
        else:
            batchu=up[batch * ubsize, um]
            batchv=vp[batch * vbsize, vm]

        batchp=[]
        batchpl=[]
        for i in range(len(labels)):
            if labels[i][0] in batchu[batch] or labels[i][1] in batchv[batch]:
                batchp.append(imgs[i])
                batchpl.append(labels[i])

        batchuv=[]
        batchm=[]
        for i in batchu[batch]:
            for j in batchv[batch]:
                batchuv.append([i,j])
                batchm.append(M[i,j])

        ufactors = []
        vfactors = []
        for i in batchu[batch]:
            ufactors.append(U[i])
        for j in batchv[batch]:
            vfactors.append(V[j])

        yield batchu,batchv,batchp,batchpl,batchuv,batchm,ufactors,vfactors
