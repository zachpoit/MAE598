import torch as t
FRAME_TIME = 0.1  # time interval
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL = 0.18  # thrust constant
state = t.tensor([2., 1., 4., 6., 2.])
action = t.tensor([3., 8.])

delta_state_gravity = -0.5 * (FRAME_TIME ** 2) * GRAVITY_ACCEL * t.tensor([0., 1., 0., 0., 0.])

print(delta_state_gravity)


state_tensor = t.zeros((1,5))
state_tensor[:,0] = -0.5 * (FRAME_TIME ** 2) * t.sin(state[4])
state_tensor[:,1] = 0.5 * (FRAME_TIME ** 2) * t.cos(state[4])
state_tensor[:,2] = -FRAME_TIME * t.sin(state[4])
state_tensor[:,3] = FRAME_TIME * t.cos(state[4])

delta_state = BOOST_ACCEL * t.mul(state_tensor,action[0])

delta_state_theta = t.mul(t.tensor([0., 0., 0., 0., 1.]), action[0])

step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0.],
                             [0., 1., 0., 0., 0.],
                             [0., 0., 1., FRAME_TIME, 0.],
                             [0., 0., 0., 1., 0.],
                             [0., 0., 0., 0., 1.]])

state = t.matmul(step_mat,state)

print(state)