# Deep-Q Network controller with LQR stabilization

This controller uses a policy trained with DQN for swinging up the pendulum and
switches to LQR for the upright stabilization once it enters the region of
attraction of the LQR controller.
