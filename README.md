# Learning multi-modal generative models with permutation-invariant encoders and tighter variational bounds


## Preliminaries

This code was developed and tested with
* Python 3.8.14
* CUDA 11.2
* JAX 0.4.3


Installation below is for CPU support only

```bash
pip install -r requirements.txt
```
# Running the scripts

#### Running the code for simulated multi-modal Gaussian data 
The aggregation flag can be SumPooling, MoE, PoE, SelfAttention, the bound flag can be masked or mixture

```bash
python gaussian_simulations.py --aggregation=SumPooling --bound=masked
```

#### Running the code for bi-modal model (continuous and categorical modality) for non-linear identifiable model
The aggregation flag can be SumPooling, MoE, PoE, SelfAttention, SumPoolingMixture, SelfAttentionMixture,
the bound flag can be masked or mixture, 
K_model flag is the number of mixtures in the prior

```bash
python iVAE_bimodal_simulation.py --aggregation=SumPooling --bound=masked --K_model=5
```


#### Running the code for multi-modal model for non-linear identifiable model
The aggregation flag can be SumPooling, MoE, PoE, SelfAttention, SumPoolingMixture, SelfAttentionMixture,
the bound flag can be masked or mixture, 
K_model flag is the number of mixtures in the prior

```bash
python iVAE_multimodal_simulation.py --aggregation=SumPooling --bound=masked --K_model=5
```