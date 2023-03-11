#!/bin/bash
#$ -l highp,h_data=2G,h_vmem=INFINITY,h_rt=23:00:00
#$ -pe shared 20
#$ -cwd
#$ -m abe

# @version      v0
# @usage        qsub human_dfe_refspectra.sh
# @description  wrapper to run python DFE1D_refspectra.py for the human dfe example
# Author: Meixi Lin
# Date: 2023-03-10 15:46:00

################################################################################
## import packages
set -eo pipefail

# CHANGE THIS TO YOUR CONDA ACTIVATION COMMAND
eval "$(/u/project/rwayne/meixilin/miniconda3/bin/conda shell.bash hook)"
conda activate dfe

################################################################################
## def variables
# CHANGE THIS TO YOUR WORKING DIRECTORY WITH THE EXAMPLE FILES
cd /u/project/klohmuel/meixilin/finwhale_DFE/scripts_varDFE/varDFE/example/human_dfe

WORKSCRIPT='../../workflow/DFE/DFE1D_refspectra.py'

################################################################################
## main
python $WORKSCRIPT \
'two_epoch' '2.332027,0.42853' '100' './output/dfe/refspectra/HS100' \
&> './output/logs/dfe_refspectra_human.log'

