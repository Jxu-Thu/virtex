import os
import moxing as mox


import argparse
parser = argparse.ArgumentParser(description="HW RUN JOB")
parser.add_argument('--gpus_per_node', action='store', help="gpu number per node")
parser.add_argument('--job_num', action='store')
# auto arg
parser.add_argument('--init_method', action='store')
parser.add_argument('--rank', action='store', default=0)
parser.add_argument('--world_size', action='store', default=1)
parser.add_argument('--data', action='store', help="The working directory.")
args, unparsed = parser.parse_known_args()

os.system('pwd') # /home/work
print('start pip install')
os.chdir('virtex-master/ViLT')
os.system('pip --no-cache-dir install -r requirements.txt')
os.system('pip --no-cache-dir install -e .')
print('finish pip install')

print('start copy dataset!')
mox.file.copy_parallel('s3://bucket-7001/luoxu/dataset/MMT/alldata', '/cache/VilT_dataset')
print('end copy dataset!')


os.system('pwd') # /home/work


Log_dir='/cache/checkpoint'
print('Run scripts!')
run_file=f'submit_hwrun{args.job_num}.sh'
strs = (f'bash submit/{run_file} {args.gpus_per_node} {args.world_size} {args.rank} {args.init_method[6:-5]} {args.init_method[-4:]}')
print(strs)
os.system(strs)


