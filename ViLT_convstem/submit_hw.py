import os
import moxing as mox
import sys


import argparse
parser = argparse.ArgumentParser(description="HW RUN JOB")
parser.add_argument('--gpus_per_node', action='store', help="gpu number per node")
parser.add_argument('--job_num', action='store')
parser.add_argument('--resume_ckpt', action='store', default=None)
parser.add_argument('--job_type', action='store', default=0) # 0 pretrain 1 finetune
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
print(f'args: {args}')


print('start copy dataset!')
if args.job_type == 0:
    mox.file.copy_parallel('s3://bucket-7001/luoxu/dataset/MMT/alldata', '/cache/VilT_dataset')
else:
    mox.file.copy_parallel('s3://bucket-7001/luoxu/dataset/MMT/all_downstream', '/cache/VilT_dataset')
print('end copy dataset!')
print_and_excute_cmd('ls /cache/VilT_dataset')

print_and_excute_cmd('pwd')

if args.resume_ckpt is not None:
    print('start copy ckpt!')
    mox.file.copy_parallel(f"s3://{args.resume_ckpt}", '/cache/checkpoint/resume_dir/last.ckpt')
    print('end copy dataset!')
    print_and_excute_cmd('ls /cache/checkpoint/resume_dir')

print('start pip install')
os.system('pip install --upgrade pip')
print('change dir to /cache/VilT_dataset/pytorch-image-models')
os.chdir('/cache/VilT_dataset/pytorch-image-models')
print_and_excute_cmd('pwd')
print_and_excute_cmd('ls')
print_and_excute_cmd('python setup.py install &>/dev/null')

print(f'os.chdir {root}/virtex-master/ViLT_convstem')
os.chdir(f'{root}/virtex-master/ViLT_convstem')
print_and_excute_cmd('pwd')
print_and_excute_cmd('ls')
os.system('pip install --ignore-installed PyYAML')
os.system('pip install -r requirements_hw.txt &>/dev/null')
# os.system('pip install pytorch_lightning==1.1.4')
# os.system('pip install torch==1.7.1')
# os.system('pip install torch==1.7.1')
# os.system('pip install transformers==4.2.1')
# os.system('pip install ipdb==0.13.4')
# os.system('pip install numpy==1.19.5')
# os.system('pip install einops==0.3.0')
# os.system('pip install pyarrow==2.0.0')
# os.system('pip install sacred==0.8.2')
# os.system('pip install pandas==1.1.5')
os.system('python setup.py install &>/dev/null')
print('finish pip install')


# print(sys.path)
# print_and_excute_cmd('which python')
# print_and_excute_cmd('which python3')


Log_dir='/cache/checkpoint'
print('Run scripts!')
if args.job_type == 0:
    run_file=f'submit_hwrun{args.job_num}.sh'
else:
    run_file = f'submit_hwfinetune{args.job_num}.sh'
strs = (f'bash submit/{run_file} {args.gpus_per_node} {args.world_size} {args.rank} {args.init_method[6:-5]} {args.init_method[-4:]}')
print(strs)
os.system(strs)


