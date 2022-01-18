#SBATCH -J test
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH --time=12:00:00
#SBATCH --partition=t4v1,p100,t4v2,rtx6000
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --export=ALL
#SBATCH --output=./test.txt
#SBATCH --gres=gpu:1
python LOLA_exact.py --n_agents 2 --print_every 1 --repeats 1 --using_nn --num_epochs 2 --init_state_representation 1   --lr_policies_inner .5 --lr_policies_outer 0.01 --base_cf_no_scale 1.6 --set_seed