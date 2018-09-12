import re

def match_pdb_file_name(file_content):
    CA_head = re.match('(\w{4}).*', file_content)
    if CA_head == None:
        return 0
    else:
        return CA_head.group(1)


import os


PDB_Path = r'F:\DNA_RNA_deeplearning\DNA_file\DNA'
os.chdir(PDB_Path)
print("you input pdb path is :", os.getcwd())
PDB_Path_listdir = os.listdir(PDB_Path)

file_w_name = r'F:\DNA_RNA_deeplearning\DNA_file\DNA.txt'
Write_file_handle = open(file_w_name, 'a')

for file_content in PDB_Path_listdir:
    file_name = match_pdb_file_name(file_content)
    if file_name:
        Write_file_handle.write(file_name)
        Write_file_handle.write(',')