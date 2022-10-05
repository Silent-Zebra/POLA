import jax.numpy as jnp

class IPD:
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    def __init__(self, init_state_coop=False, contrib_factor=1.33):
        cc = contrib_factor - 1.
        dd = 0.
        dc = contrib_factor / 2. # I defect when opp coop
        cd = contrib_factor / 2. - 1 # I coop when opp defect
        self.payout_mat = jnp.array([[dd, dc],[cd, cc]])
        # One hot state representation because this would scale to n agents
        self.states = jnp.array([[[1, 0, 0, 1, 0, 0], #DD (WE ARE BACK TO THE REPR OF FIRST AGENT, SECOND AGENT)
                                          [1, 0, 0, 0, 1, 0]], #DC
                                         [[0, 1, 0, 1, 0, 0], #CD
                                          [0, 1, 0, 0, 1, 0]]]) #CC
        if init_state_coop:
            self.init_state = jnp.array([0, 1, 0, 0, 1, 0])
        else:
            self.init_state = jnp.array([0, 0, 1, 0, 0, 1])

    def reset(self, unused_key):
        return self.init_state, self.init_state

    def step(self, unused_state, ac0, ac1, unused_key):

        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        state = self.states[ac0, ac1]
        observation = state
        reward = (r0, r1)
        # State is observation in the IPD
        return state, observation, reward, None
