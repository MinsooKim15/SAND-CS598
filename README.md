
# SAnD implementation CS598

This is the implementation of SAnD which is the attention based approach on EHR data proposed in the paper "Attend and Diagnose: Clinical Time Series Analysis using Attention Models."(https://arxiv.org/abs/1711.03905)

And this implementation is part of final project of CS598 Deep Learning for HealthCare class in UIUC.


## Installation

This project requires followings.

- PyTorch
- Pandas
- Scikit learn

```bash
  https://github.com/MinsooKim15/SAND-CS598.git
```

Also this code includes implementation of SAnD and have benchmark test on MIMIC-3 data. This project does not include MIMIC-3. It is using following github project "https://github.com/YerevaNN/mimic3-benchmarks".

To do benchmark test, you should also add submodule of following MIMIC-3 benchmark.

```bash
    git submodule add https://github.com/YerevaNN/mimic3-benchmarks.git
```

As this repository does not include the MIMIC3-data which is very sensitive health-care data, to do benchmark test you need to download the MIMIC-3 csv file and clear them following the instruction from the mimic3-benchmarks repository.
## Features

- In-hospital-mortality benchmark.

To do this benchmark task, you can run in command-line.

```bash
python main/test.py test_in_hospital_mortality --n_heads 64 --factor 24 --num_class 3 --num_layers 6 --learning_rate 0.001
```


## Appendix

Any additional information goes here


## Acknowledgements
This is the implementation of the paper "Attend and Diagnose: Clinical Time Series Analysis using Attention Models"
 - [Attend and Diagnose: Clinical Time Series Analysis using Attention Models](https://arxiv.org/abs/1711.03905)

As the most of structure came from attention and transformer apporoach, following paper was key to implementation.
 - [Attention is all you need](https://arxiv.org/abs/1706.03762)


This project's implementation was done from the great implementation by following repository. 
 - [khirotaka/SAnD](https://github.com/khirotaka/SAnD)


Using the MIMIC-3 from scratch would be impossible without this project 
 - [YerevaNN/mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks)

