import h5py
import os
import numpy as np
import xlrd

#位置21
Hydrophilicity_value = {'A':-0.5,'L':-1.8,'R':3.0,'K':3.0,'N':0.2,'M':-1.3,'D':3.0,'F':-2.5,'C':-1.0,'P':0,'Q':0.2,'S':0.3,'E':3.0,'T':-0.4,'G':0,'W':-3.4,'H':-0.5,'Y':-2.3,'I':-1.8,'V':-1.5}

#位置22
Hydrophobicity_value = {'A':-0.21,'L':-4.68,'R':2.11,'K':3.88,'N':0.96,'M':-3.66,'D':1.36,'F':-4.65,'C':-6.04,'P':0.75,'Q':1.52,'S':1.74,'E':2.30,'T':0.78,'G':0,'W':-3.32,'H':-1.23,'Y':-1.01,'I':-4.81,'V':-3.5}

#位置23
charge_value = {'A':0,'L':0,'R':1,'K':1,'N':0,'M':0,'D':-1,'F':0,'C':0,'P':0,'Q':0,'S':0,'E':-1,'T':0,'G':0,'W':0,'H':0,'Y':0,'I':0,'V':0}

#位置24
hydrogen_value = {'A':0,'L':0,'R':4,'K':2,'N':2,'M':0,'D':1,'F':0,'C':0,'P':0,'Q':3,'S':2,'E':4,'T':2,'G':0,'W':0,'H':1,'Y':2,'I':0,'V':0}

#位置0
aatype_value = {'A':1,'L':2,'R':3,'K':4,'N':5,'M':6,'D':7,'F':8,'C':9,'P':10,'Q':11,'S':12,'E':13,'T':14,'G':15,'W':16,'H':17,'Y':18,'I':19,'V':20}


#氨基酸名称转换
AAtype_name_tran = {'GLY':'G','ALA':'A','VAL':'V','LEU':'L','ILE':'I','PHE':'F','TRP':'W','TYR':'Y','ASP':'D','HIS':'H','ASN':'N','GLU':'E','LYS':'K','GLN':'Q','MET':'M','ARG':'R','SER':'S','THR':'T','CYS':'C','PRO':'P'}




def check_aa_site_state(pssm_name,aatype,chain_type,chain_num):
    AS = 0
    site_Path_listdir = os.listdir(site_Path)
    for site_flie_name in site_Path_listdir:
        if site_flie_name[0:4].upper() == pssm_name:
            data = xlrd.open_workbook(site_flie_name)
            table = data.sheets()[0]
            nrows = table.nrows
            interface_counter = 0
            NA_type = 0
            for i in range(nrows):
                ATOM_cell = table.row_values(i)
                if ATOM_cell[1] in AAtype_name_tran and \
                    ATOM_cell[2] == chain_type and \
                    int(ATOM_cell[3]) == chain_num:
                    AS = float(ATOM_cell[9])
                    if ATOM_cell[7]:
                        return [1,AS]
    return [0,AS]



pssm_Path = r"F:\DNA_RNA_deeplearning\DNA_file\新建文件夹"
os.chdir(pssm_Path)
print("the pssm path：",pssm_Path)

site_Path = r"F:\DNA_RNA_deeplearning\DNA_file\DNA_EXCEL"
print("the site path：",site_Path)

pssm_Path_listdir = os.listdir(pssm_Path)



for pssm_flie_name in pssm_Path_listdir:
    pssm_file_handle = open(pssm_flie_name, 'r')
    content_flag = 0
    print(pssm_flie_name)
    counter = 0


    data = np.zeros((20000, 27))
    drug_name_str = 0

    for pssm_file_content in pssm_file_handle:
        #检测pssm矩阵结束
        if content_flag:
            if pssm_file_content[162:163] != '.':
                content_flag = 0

        #提取pssm矩阵中单的信息
        if content_flag:
            col = 0

                #剩余匹配位点信息
            if pssm_file_content[6:7] in aatype_value:
                data[counter, 21] = Hydrophilicity_value[pssm_file_content[6:7]]
                data[counter, 22] = Hydrophobicity_value[pssm_file_content[6:7]]
                data[counter, 23] = charge_value[pssm_file_content[6:7]]
                data[counter, 24] = hydrogen_value[pssm_file_content[6:7]]
                os.chdir(site_Path)
                site_flag,data[counter, 25] = check_aa_site_state(pssm_flie_name[0:4],pssm_file_content[6:7],pssm_flie_name[5:6],int(pssm_file_content[0:5]))
                if site_flag:
                    # drug_name_str = check_aa_site_state(pssm_flie_name[0:4], pssm_file_content[6:7], pssm_flie_name[5:6],int(pssm_file_content[0:5]))
                    data[counter, 26] = 1

                os.chdir(pssm_Path)

            else:
                data[counter, 21] = 0
                data[counter, 22] = 0
                data[counter, 23] = 0
                data[counter, 24] = 0


            while col < 21:
                if col:
                    data[counter,col] = int(pssm_file_content[9+((col-1)*3):12+((col-1)*3)])
                else:
                    if pssm_file_content[6:7] != 'U'and \
                            pssm_file_content[6:7] != 'X':
                        data[counter,0] = aatype_value[pssm_file_content[6:7]]
                    else:
                        data[counter,0] = 0
                col = col + 1
            counter = counter + 1

        #检测pssm矩阵开始的位置
        if pssm_file_content[11:12] == 'A':
            content_flag = 1

    f=h5py.File(r"F:\DNA_RNA_deeplearning\DNA_file\DNA_h5py\\"+pssm_flie_name[0:6]+".hdf5","w")
    f.create_dataset('train', data = data[:counter,:])









#
# data = np.zeros((100,200))
# print(data)
#
#
# f=h5py.File("myh5py.hdf5","w")
# #deset1是数据集的name，（20,）代表数据集的shape，i代表的是数据集的元素类型
#
# f.create_dataset('train_set_x', data = data)
#
#
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].shape)
#     print(f[key].value)