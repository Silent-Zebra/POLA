# Proximal Learning with Opponent Learning Awareness (POLA)

## Sample Commands used for Figure 1

I give a few examples here. There are a LOT for the plot, so here are just a few that tackle harder problem settings (the easier problem settings can use similar settings, otherwise it's not too hard to tune them). Init state representation 2 means the start state is different from the other states (0 = defect, 1 = cooperate)

### LOLA Func Approx

CF=1.33

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.05 --actual_update --base_cf_no_scale 1.33

CF=1.4

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer 0.1 --actual_update --base_cf_no_scale 1.4

### (Outer) POLA Func Approx

CF=1.1

python LOLA_exact.py --n_agents 2 --using_nn --print_every 1 --repeats 20 --num_epochs 2 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.03 --outer_exact_prox --outer_beta 0.1 --actual_update --base_cf_no_scale 1.1


## Commands used for Figure 2

### LOLA:

Neural Net 1:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 1 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .05 --actual_update --base_cf_no_scale 1.33

Neural Net 2:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 4 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .05 --actual_update --base_cf_no_scale 1.33

Tabular:

python LOLA_exact.py --n_agents 2 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --std 0  --lr_policies_inner 1 --lr_policies_outer .2 --set_seed --seed 1 --actual_update --base_cf_no_scale 1.33

### POLA:

Neural Net 1:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 1 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.003 --lr_policies_outer .003 --actual_update --base_cf_no_scale 1.33 --inner_exact_prox --inner_beta 2 --outer_exact_prox --outer_beta 5 --print_prox_loops_info

Neural Net 2:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 4 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.003 --lr_policies_outer .003 --actual_update --base_cf_no_scale 1.33 --inner_exact_prox --inner_beta 2 --outer_exact_prox --outer_beta 5 --print_prox_loops_info

Tabular:

python LOLA_exact.py --n_agents 2 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --std 0  --lr_policies_inner 0.1 --lr_policies_outer .1 --set_seed --seed 1 --actual_update --base_cf_no_scale 1.33 --inner_exact_prox --inner_beta 2 --outer_exact_prox --outer_beta 5 --print_prox_loops_info




