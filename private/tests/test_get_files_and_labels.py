from cenotaph.basics.generic_functions import get_files_and_folders
from cenotaph.basics.generic_functions import get_files_in_folder

root = '../data/testdir'
files, labels = get_files_and_folders(root)
for i in range(0,len(files)):
    print('%s \t %s' %(files[i], labels[i]))
    
print('Files: ' + str(files))
print(len(files))
