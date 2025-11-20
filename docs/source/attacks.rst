Attacks Overview
================

This page provides a theoretical overview of the adversarial attack methods implemented in the ``mlff_attack`` package.

FGSM
----

The Fast Gradient Sign Method (FGSM) attack was first implemeted by Goodfellow et al. in the 2014 paper `Explaining and Harnessing Adversarial Examples <https://arxiv.org/abs/1412.6572>`_.

The update rule for FGSM is 

.. math::
   x^* = x + \epsilon \cdot sign(\nabla_x \mathcal{L}(x))


where :math:`x` are the original atomic positions, :math:`\epsilon` is the perturbation magnitude, :math:`\nabla_x \mathcal{L}(x)` is the gradient of the loss function with respect to the atomic positions, and :math:`x^*` are the perturbed atomic positions after the attack.

I-FGSM
------

Iterative-FGSM, also known as BIM or "Basic Iterative Method", repeatedly updates the atomic positions using FGSM over several time steps.

.. math::
   x^{(t+1)} = Clip_{x, \epsilon} \{ x^{(t)} + \alpha \cdot sign(\nabla_x \mathcal{L}(x^{(t)})) \}

where :math:`x^{(t)}` are the atomic positions at iteration :math:`t`, :math:`\alpha` is the step size for each iteration, and :math:`Clip_{x, \epsilon}` ensures that the perturbations remain within the specified :math:`\epsilon`-ball around the original positions.

If the ``clip`` option is set to ``False``, the clipping step is omitted, allowing the perturbations to exceed the :math:`\epsilon` limit.

This method was introduced in 2016 by Kurakin et al. in `Adversarial Examples in the Physical World <https://arxiv.org/abs/1607.02533>`_.

PGD
---

PGD begins by randomly initializing the perturbation. It then applies the following update rule, similar to FGSM and I-FGSM

.. math::
   x^{(t+1)} = \Pi_{x, \epsilon} \{ x^{(t)} + \alpha \cdot sign(\nabla_x \mathcal{L}(x^{(t)})) \}

where :math:`\Pi_{x, \epsilon}` is the projection operator that ensures the perturbed positions remain within the :math:`\epsilon`-ball around the original positions.

PGD outperforms FGSM and I-FGSM because the random start helps to escape local minima, making it a stronger adversarial attack. 

This method was proposed by Madry et al. in `Towards Deep Learning Models Resistant to Adversarial Attacks <https://arxiv.org/abs/1706.06083>`_.