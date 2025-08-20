https://arxiv.org/pdf/2104.13906

4.1 Identifying unsafe reward shaping

staying close to the center of the lane [23], 
passing other vehicles [31],
not changing lanes [20], 
increasing distances from other vehicles [51]
avoiding overlap with the opposite-direction lane [12, 28]
and steering straight at all times

reducing gas cost - could discourage accelerating, even when doing so would avoid a potential collision
avoiding collisions

reducing a correctly weighted gas cost would presumably have negligible effect on the frequency of collisions, since
the benefit of avoiding collisions would far outweigh the benefit of reducing gas cost.

4.2 Comparing preference orderings

4.4 Identifying learnable loopholes
Once a designed reward function is used for learning a policy, observable
patterns of undesirable behavior might emerge. When such behavior increases
utility, it is often referred to as reward hacking or specification gaming (terms
that implicitly and unfairly blame the agent for correctly optimizing a flawed
utility function). Colloquially, such technically legal violations of the spirit of
the law are called loopholes. 



1. the California Department of Motor Vehicles (DMV) requires reporting of miles per disengagement, where a disengagement is defined as the deactivation of the vehicle’s autonomous mode and/or a safety driver taking control from the autonomous system.
2. The California DMV also requires a report to be filed for each collision, through which a miles per collision measure is
sometimes calculated
3.  miles per fatality
4. progress to the destination, obeying the law, and passenger experience
(route length from position at time t) − (route length from position at time t − 1)
-----------------------------------------------------------------------------------
                                  route length from start

passenger experience: many people have experienced fear as passengers when their driver brakes later than they prefer, creating a pre-braking moment of uncertainty regarding whether the driver is aware of the need to slow the vehicle, which they are.

The reward function is commonly a linear combination of attributes; however, this linearity
assumption could be incorrect, for example if the utility function needs to be the result of a conjunction over attributes,
such as a binary utility function for which success is defined as reaching the destination without collision. Also, weight
assignment for such linearly expressed reward functions is often done by trial and error (see Section 4.5), possibly in
part because the researchers lack a principled way to weigh attributes with different units.

Utilities expressed in currency units are common in RL when profit or cost
reduction are the explicit goals of the task, such as stock trading [33] and tax collection [30], but we are unaware of its
usage as an optimization objective for AD.


To create such a financial utility function for AD, non-financial outcomes would need to be mapped to financial values,
perhaps via an assessment of people’s willingness to pay for those outcomes. We have been surprised to find that
some non-financial outcomes of driving have a more straightforward financial expression than we initially expected,
providing optimism for this strategy of reward design. For example, much effort has gone towards establishing a
value of statistical life, which allows calculation of a monetary value for a reduction in a small risk of fatalities. The
value of statistical life is used by numerous governmental agencies to make decisions that involve both financial costs
and risks of fatality. The US Department of Transportation’s value of statistical life was $11.6 million US Dollars in
2020 [49, 48].


https://openreview.net/pdf?id=JYtwGwIL7ye
True reward: minimize the mean commute
Proxy reward: increase the mean velocity (mean velocity of all cars - RL + human)
When it starts merging, it realizes it would slow the other cars. But the mean commute time increase since the red car is stuck.  An RL agent regulating traffic learns to stop any cars from merging
onto the highway in order to maintain a high average velocity of the cars on the straightaway.

@inproceedings{
    pan2022rewardhacking,
    title={The Effects of Reward Misspecification: Mapping and Mitigating Misaligned Models},
    author={Alexander Pan and Kush Bhatia and Jacob Steinhardt},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=JYtwGwIL7ye}
}

"Figure 1: An example of reward hacking when cars merge onto a highway. A human-driver model
controls the grey cars and an RL policy controls the red car. The RL agent observes positions
and velocities of nearby cars (including itself) and adjusts its acceleration to maximize the proxy
reward. At first glance, both the proxy reward and true reward appear to incentivize fast traffic flow.
However, smaller policy models allow the red car to merge, whereas larger policy models exploit
the misspecification by stopping the red car. When the red car stops merging, the mean velocity
increases (merging slows down the more numerous grey cars). However, the mean commute time
also increases (the red car is stuck). This exemplifies a phase transition: the qualitative behavior of
the agent shifts as the model size increases."

We focus on the Traffic-Mer environment from Figure 2a, where minimizing
average commute time is replaced by maximizing average velocity. In this case, smaller policies
learn to merge onto the straightaway by slightly slowing down the other vehicles (Figure 4a). On the
other hand, larger policy models stop the AVs to prevent them from merging at all (Figure 4b). This
increases the average velocity, because the vehicles on the straightaway (which greatly outnumber
vehicles on the on-ramp) do not need to slow down for merging traffic. However, it significantly
increases the average commute time, as the passengers in the AV remain stuck


