# Proximal Learning with Opponent Learning Awareness (POLA)

## Sample Commands used for Figure 1

I give a few examples here. There are a LOT for the plot, so here are just a few that tackle harder problem settings (the easier problem settings can use similar settings, otherwise it's not too hard to tune them)

### LOLA Func Approx

CF=1.33
python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.05 --actual_update --base_cf_no_scale 1.33

CF=1.4
python LOLA_exact.py --n_agents 2 --using_nn --nn_hidden_size 16 --print_every 20 --repeats 20 --num_epochs 100 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.2 --lr_policies_outer 0.1 --actual_update --base_cf_no_scale 1.4

### POLA Func Approx

CF=1.1
python LOLA_exact.py --n_agents 2 --using_nn --print_every 1 --repeats 20 --num_epochs 2 --init_state_representation 2  --set_seed --seed 1 --lr_policies_inner 0.3 --lr_policies_outer 0.03 --outer_exact_prox --outer_beta 0.1 --actual_update --base_cf_no_scale 1.1
