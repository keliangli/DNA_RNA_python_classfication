import h5py
import os
import numpy as np



h5py_Path = r"F:\DNA_RNA_deeplearning\RNA_file\RNA_hp5y"

os.chdir(h5py_Path)

print("the pssm path：",h5py_Path)

h5py_Path_listdir = os.listdir(h5py_Path)

a = np.zeros((25,26))

for h5py_file_name in h5py_Path_listdir:
    f = h5py.File(h5py_file_name, 'r')
    h5py_data = f['train']
    print(h5py_file_name)
    if h5py_data.shape[0]>24:
        data_x = np.zeros(((h5py_data.shape[0]-24),25,26))
        data_y = np.zeros(h5py_data.shape[0]-24)
        counter = 12
        while(counter < (h5py_data.shape[0]-12)):
            data_x[counter-12,:,:] = h5py_data[counter-12:counter+13,:-1]
            a = h5py_data[counter-12:counter+13,:-1]
            data_y[counter-12]     = h5py_data[counter,-1]
            counter = counter + 1
    row_num = 0

    p_data_x = np.zeros((1,25,26))
    p_data_y = np.zeros((1))
    n_data_x = np.zeros((1,25,26))
    n_data_y = np.zeros((1))
    while row_num < data_y.shape[0]:
        if data_y[row_num]:
            p_data_x= np.append(p_data_x,data_x[row_num:row_num+1,:,:], axis = 0)
            p_data_y= np.append(p_data_y,data_y[row_num:row_num+1], axis=0)
        else:
            n_data_x= np.append(n_data_x,data_x[row_num:row_num+1,:,:], axis = 0)
            n_data_y= np.append(n_data_y,data_y[row_num:row_num+1], axis=0)
        row_num = row_num + 1
    p_data_x = p_data_x[1:,:,:]
    p_data_y = p_data_y[1:]
    n_data_x = n_data_x[1:,:,:]
    n_data_y = n_data_y[1:]

    p_row_num = p_data_y.shape[0]
    # print(p_data_y)

    p_data_x = np.append(p_data_x, n_data_x[:p_row_num,:,:], axis=0)
    p_data_y = np.append(p_data_y, n_data_y[:p_row_num], axis=0)

    # print(p_data_y)

    if p_row_num :
        f=h5py.File("25x25_"+h5py_file_name,"w")
        f.create_dataset('train_x', data = p_data_x)
        f.create_dataset('train_y', data = p_data_y)


