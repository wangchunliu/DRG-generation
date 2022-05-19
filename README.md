# DRG-generation
This is the repository for the DRG-to-text generation task.
The related paper is "----".
(i) We present the preprocess for converting DRS data to DRG data.
(ii) We use graph neural networks (GNNs) to do the generation.
(iii) We propose deep traversal encoder.
This project is implemented using the framework OpenNMT-py.
## Environments and Dependencies
- python 3.6
- PyTorch 1.5.0

## Datasets
In our experiments, we use the PMB version 4.0.0 SBN data, you can download from https://pmb.let.rug.nl/data.php, where include gold, silver and bronze data.
For test and development set, you can randomly split gold data, or you can use the gold data I alrighdy uploaded in our repository.
.
## Preprocess
You may need some preprocess scripts to convert the dataset into the format required for the model, please see the folder of [data_preprocess]()
We use [Moses tokenizer] (www.statmt.org/moses/), and automatic evaluation metrics from https://github.com/tuetschek/e2e-metrics.


