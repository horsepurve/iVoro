# iVoro
Official code for **Progressive Voronoi Diagram Subdivision Enables Accurate Data-free Class-Incremental Learning**, ICLR'23.

### Introduction
The iVoro method is based on the idea of Voronoi Diagram subdivision from Computational Geometry.
<p align="center">
  <img src="./img/iVoro-fig1.PNG" width="450">
</p>

**A**. Establish Voronoi Diagram based on base model.
**B**. Insertion of a new class as a new Voronoi cell enables the minimal intervention to the overall structure.
**C**. Divide-and-conquer (a classical algorithm for Voronoi construction) efficiently introduce a batch of new classes into the system.


### Results
The results of MNIST in 2D space below clearly showed different space subdivision results from conventional fine-tuning, PASS, and different variants of iVoro.
<p align="center">
  <img src="./img/iVoro-fig2.PNG" width="750">
</p>

### Reproducing the results

Step 1. Training of the base model, please follow PASS (github).

Step 2. Download the feature files.
Google Drive

Go to the directory:
```bash
cd MNIST
```

### Reference
[ICLR'23](https://openreview.net/forum?id=zJXg_Wmob03)

### Acknowledgments

### Contact
If you have any problem please [contact me](mailto:horsepurve@gmail.com).
