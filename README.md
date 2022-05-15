# Proximal Learning with Opponent Learning Awareness (POLA)

## Example Commands used for Figure 1

I give a few examples here. There are a LOT for the plot, so here are just a few that tackle harder problem settings (the easier problem settings can use similar settings, otherwise it's not too hard to tune them. I don't give hyperparameters on the harder settings where as I tried several configurations but couldn't get any combination of hyperaparameters to find TFT with probability > 0). Init state representation 2 means the start state is different from the other states (0 = defect, 1 = cooperate)

### LOLA Tabular

CF=1.1

python LOLA_exact.py --n_agents 2 --print_every 10 --repeats 20 --num_epochs 30 --init_state_representation 2 --set_seed --seed 1 --lr_policies_inner 3 --lr_policies_outer 50 --actual_update --base_cf_no_scale 1.1

### LOLA Func Approx

CF=1.33

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.05 --actual_update --base_cf_no_scale 1.33

CF=1.4

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer 0.1 --actual_update --base_cf_no_scale 1.4

CF=1.6

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2 --set_seed --seed 1 --lr_policies_inner 0.1 --lr_policies_outer 0.05 --actual_update --base_cf_no_scale 1.6


### (Outer) POLA Func Approx

CF=1.1

python LOLA_exact.py --n_agents 2 --using_nn --print_every 1 --repeats 20 --num_epochs 2 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.03 --outer_exact_prox --outer_beta 0.1 --actual_update --base_cf_no_scale 1.1

### (Outer) POLA Pre-Conditioned Tabular

CF=1.25

python LOLA_exact.py --n_agents 2 --ill_condition --print_every 5 --repeats 20 --num_epochs 10 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.05 --outer_exact_prox --outer_beta 0.5 --actual_update --base_cf_no_scale 1.25

CF=1.33

python LOLA_exact.py --n_agents 2 --ill_condition --print_every 5 --repeats 20 --num_epochs 10 --init_state_representation 2  --set_seed --seed 2 --lr_policies_inner 0.3 --lr_policies_outer 0.05 --outer_exact_prox --outer_beta 0.5 --actual_update --base_cf_no_scale 1.33

## Commands used for Figure 2

### LOLA:

Neural Net 1:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 1 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .05 --actual_update --base_cf_no_scale 1.33

Neural Net 2:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 4 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .05 --actual_update --base_cf_no_scale 1.33


### POLA:

Neural Net 1:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 1 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2 --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .005 --actual_update --base_cf_no_scale 1.33 --outer_exact_prox --outer_beta 5 --print_prox_loops_info

Neural Net 2:

python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 2 --custom_param 4 --print_every 1 --repeats 1 --num_epochs 1 --init_state_representation 2 --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer .005 --actual_update --base_cf_no_scale 1.33 --outer_exact_prox --outer_beta 5 --print_prox_loops_info

## Commands used for Figure 3

Change the --seed parameter for different seeds.

### LOLA-DiCE

python LOLA_dice.py  --env ipd  --val_update_after_loop --init_state_coop  --n_update 15000 --gae_lambda 1.0  --inner_steps 2 --outer_steps 1 --lr_in 0.03 --lr_out 0.005 --lr_v 0.002 --batch_size 4000 --len_rollout 50  --print_every 50 --checkpoint_every 3000 --seed 3

### POLA-DiCE

python LOLA_dice.py  --env ipd  --val_update_after_loop --init_state_coop  --n_update 200 --gae_lambda 1.0  --inner_steps 1 --outer_steps 100 --lr_in 0.03 --lr_out 0.005 --lr_v 0.002 --batch_size 4000 --len_rollout 50  --print_every 1 --outer_beta 4 --checkpoint_every 50 --seed 3

### POLA-OM

python LOLA_dice.py  --env ipd  --val_update_after_loop --init_state_coop  --n_update 200 --gae_lambda 1.0  --inner_steps 1 --outer_steps 100 --lr_in 0.03 --lr_out 0.005 --lr_v 0.002 --batch_size 4000 --len_rollout 50  --print_every 1 --outer_beta 4 --checkpoint_every 50 --seed 3 --opp_model --opp_model_steps 1 --opp_model_data_batches 100 --om_lr_p 0.005 --om_lr_v 0.002

## Commands used for Figure 4

Change the --seed parameter for different seeds.

### LOLA-DiCE

python LOLA_dice.py  --env ogcoin --val_update_after_loop   --n_update 15000 --gae_lambda 1.0  --inner_steps 2 --outer_steps 1 --lr_in 0.03 --lr_out 0.005 --lr_v 0.00005 --batch_size 4000 --len_rollout 50  --print_every 50 --checkpoint_every 3000 --seed 1 

### POLA-DiCE

python LOLA_dice.py  --env ogcoin --val_update_after_loop   --n_update 200 --gae_lambda 1.0  --inner_steps 1 --outer_steps 100 --lr_in 0.03 --lr_out 0.005 --lr_v 0.00005 --batch_size 4000 --len_rollout 50  --print_every 1 --outer_beta 4 --checkpoint_every 50 --seed 10 

### POLA-OM

python LOLA_dice.py --env ogcoin --val_update_after_loop --zero_vals --n_update 200 --gae_lambda 1.0  --inner_steps 1 --outer_steps 100 --lr_in 0.03 --lr_out 0.005 --lr_v 0.0 --batch_size 4000 --len_rollout 50  --print_every 1 --outer_beta 4 --checkpoint_every 25  --seed 8 --opp_model --opp_model_steps 1 --opp_model_data_batches 200 --om_lr_p 0.005 --om_lr_v 0.0

