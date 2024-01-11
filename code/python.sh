#!/bin/bash

### python.sh
###########################################################################
## environment & variable setup
####### job customization
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 8:00:00
#SBATCH -p a100_normal_q
#SBATCH -A ece6524-spring2023
#SBATCH --gres=gpu:1
####### end of job customization
# end of environment & variable setup
###########################################################################
#### add modules:

#end of add modules
###########################################################################
###print script to keep a record of what is done
cat python.sh
echo "python code"
###########################################################################
echo start training
python training.py

exit;