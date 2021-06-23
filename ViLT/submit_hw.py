import os
import moxing as mox
import sys


import argparse
parser = argparse.ArgumentParser(description="HW RUN JOB")
parser.add_argument('--gpus_per_node', action='store', help="gpu number per node")
parser.add_argument('--job_num', action='store')
# auto arg
parser.add_argument('--init_method', action='store')
parser.add_argument('--rank', action='store', default=0)
parser.add_argument('--world_size', action='store', default=1)
args, unparsed = parser.parse_known_args()

def print_and_excute_cmd(str):
    print(str)
    os.system(str)
    
root=os.path.abspath('.')
print(f'root: {root}')
print_and_excute_cmd('args')
print_and_excute_cmd('pwd')
print_and_excute_cmd('ls')
print_and_excute_cmd(f'cd {root}/virtex-master/ViLT')
print_and_excute_cmd('pwd')
print_and_excute_cmd('ls')

print('start pip install')
os.system('pip install --upgrade pip')
os.system('pip install --ignore-installed PyYAML')
os.system('pip install -r requirements_hw.txt')
os.system('pip install -e .')
print('finish pip install')


print(sys.path)
print('check python path')
os.system('which python') # /home/work

print('start copy dataset!')
mox.file.copy_parallel('s3://bucket-7001/luoxu/dataset/MMT/alldata', '/cache/VilT_dataset')
print('end copy dataset!')
os.system('ls /cache/VilT_dataset') # /home/work


os.system('pwd') # /home/work


Log_dir='/cache/checkpoint'
print('Run scripts!')
run_file=f'submit_hwrun{args.job_num}.sh'
strs = (f'bash submit/{run_file} {args.gpus_per_node} {args.world_size} {args.rank} {args.init_method[6:-5]} {args.init_method[-4:]}')
print(strs)
os.system(strs)


