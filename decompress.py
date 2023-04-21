import os
import lzip

os.chdir("/Users/horvathbalint/Data/Options/Raw_SAS")


list_of_compressed_files = ['ticker.sas7bdat.lz']

for file in list_of_compressed_files:
    with open(file[:-3], 'wb') as sink:
        for chunk in lzip.decompress_file_iter(file):
            sink.write(chunk)
    print('Decompressed file {}'.format(file))







