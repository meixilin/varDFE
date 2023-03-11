# installing the `varDFE` package

## recommended

0. install `miniconda3`
1. set up the conda environment and install the required packages

```bash
conda env create -f dfe.yml
```

dfe recipe (`dfe.yml`)

```yaml
name: dfe
channels:
  - conda-forge
  - defaults
  - bioconda
dependencies:
  - dadi=2.1.1=py310h2132740_4
  - ipykernel=6.9.1=py310hecd8cb5_0
  - biopython=1.79=py310he24745e_1
  - pandas=1.4.1=py310he9d5cce_1
  - scikit-allel=1.3.5=py310hdd25497_1
  - plotnine=0.8.0=pyhd8ed1ab_0
  - mpmath=1.2.1
  - pyinstaller=5.0=py310he04095b_0
```

2. install `varDFE` as an editable package

```bash
cd <your-dir>/varDFE
ls
# README.md example pyproject.toml setup.cfg src workflow
conda activate dfe
which pip
# should be: <your-dir>/miniconda3/envs/dfe/bin/pip
pip install -e ./
```


