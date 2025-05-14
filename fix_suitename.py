import os
import os.path

import shutil

from pathlib import Path


dir_path='./helm_output/eval_1000/'

files = [os.path.join(dirpath, filename)
    for (dirpath, dirs, files) in os.walk(dir_path)
    for filename in (dirs + files)]

files = [file for file in files if os.path.isfile(file)]

for file in files:
    new_path=file
    src=file
    new_path=new_path.replace('eval_1000', 'full_wmt_1_samples_1000_evals')
    new_path=new_path.replace('/wmt/', '/wmt_14_language_pair_de_en_/')
    new_path=new_path.replace('/gsm/', '/gsm_/')

    if not os.path.isfile(new_path):
        print(new_path)
        # new_dir=os.path.dirname(new_path)
        # Path(new_dir).mkdir(parents=True, exist_ok=True)
        # shutil.copyfile(src, new_path)


