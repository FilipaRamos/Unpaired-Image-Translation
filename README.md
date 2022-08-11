# Unpaired-Image-Translation

Welcome to this repository! This is my try at generating Monet paintings from photographs through unpaired image-to-image translation. The model is a variation of the base Cycle-GAN and the data used comes from Kaggle's challenge "I'm something of a painter myself".

![Generated Paintings](https://github.com/FilipaRamos/Unpaired-Image-Translation/blob/main/resources/cover_monet_translated.png)

I developed this code for a Machine Learning course in the scope of my PhD studies. The abstract of the work I carried out is as follows:


> The field of painting generation is a subset of image-to-image translation problems where the goal is to map an input image to an output image without presence of paired data. Focusing on the specific translation of photographs into Monet paintings, this work analyses the potentials of CycleGAN on these specific domains. Some improvements to the base model are proposed, including an encoder with shared weights, a tuned architecture and an asymmetry enforcing parameter. These configurations are explored using both qualitatively and quantitatively evaluation methodologies, showcasing significantly better results than the initial baseline.

#### Execution

Options that can be passed to the training script can be found in the file `main.py`.

``python main.py --epochs 50``