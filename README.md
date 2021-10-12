# SemSegAdvPatch
This is the code repository for the paper "*Evaluating the Robustness of Semantic Segmentation for Autonomous Driving against Real-World Adversarial Patch Attacks*" by Federico Nesti, Giulio Rossolini, Saasha Nair, Alessandro Biondi, and Giorgio Buttazzo. The paper was accepted for WACV 2022. A [pre-print](https://arxiv.org/abs/2108.06179) is also available.

In this paper we extensively explore the robustness of real-time Semantic Segmentation models against adversarial patches in the context of autonomous driving. We perform experiments both on Cityscapes and on datasets collected with the CARLA simulator.
We propose a novel loss function that improves attack performance. We also propose the scene-specific attack, a novel attack methodology that exploits geometrical information from the simulator to apply accurately re-projected patches onto a known, fixed attackable surface (i.e., a billboard). We found that this attack outperforms the standard Expectation-Over-Transformation attack methodology.

The networks under tests proved to be somehow robust to these kinds of attacks, hence raising interesting questions not only about the actual real-world adversarial attack effect, but also on the intrinsic robustness of semantic segmentation models.


https://user-images.githubusercontent.com/92364988/136968170-8297abfa-392a-4750-a448-dfcabc57ef62.mp4


## Code
The code is coming soon with additional features!
