# PyMC RBA
Direct PyMC implementation of AFNI's region Based analysis (RBA) analysis.

The following repo is inspired by a project that I was privy too at OHBM hackathon 2022 (at time did not have the technical skills to assist that much with) that was inspired by the the AFNI group reasearcher Gang chen who implemented Bayesian multilevel modelling for region based analysis with the use Brms Bürkner (2021) and the [RBA.R programme](https://github.com/afni/afni/blob/master/src/R_scripts/RBA.R) who initially wanted to implement the programme through PyBrms [see](https://github.com/adamhaber/pybrms). Unforutnately their is no active support of Pybrms. As such the project was taken away by a group of researchers and taken to end state using a PyMC-labs Bambi (Abril-Pla et al. 2023) implementation. This allowed for an ease of use to the end users and the option for variety of modern inference algorithms gpu utitlities with use of Googles jax auto-diff library. Through use of four Nvidia Tesla V100 GPUs. they got signifcant speed ups (Moia et al., 2024)

What does this repo offer then? The following code is direct implementaion of RBA type models in raw pymc. This will allow for learning for end users how implement such methods directly and the freedom this can provide as well as. Additionally speed ups can be achived with direct implementation that is not possible when using generalised data analsysis softwares such as brms.

Note: The author is a massive fan of Bambi package. The resources provided by the PyMC-lab/Bambi team made these speedups possible throught the open sourcing of their code. For that I am grateful.

# References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L., Fonnesbeck, C. J., Kochurov, M., ... & Zinkov, R. (2023). PyMC: a modern, and comprehensive probabilistic programming framework in Python. PeerJ Computer Science, 9, e1516.

Chen, G., Xiao, Y., Taylor, P. A., Rajendra, J. K., Riggins, T., Geng, F., ... & Cox, R. W. (2019). Handling multiplicity in neuroimaging through Bayesian lenses with multilevel modeling. Neuroinformatics, 17, 515-545.

Moia, S., Wang, H. T., Heinsfeld, A. S., Jarecka, D., Yang, Y. F., Heunis, S., ... & Bijsterbosch, J. D. (2024). Proceedings of the OHBM Brainhack 2022.