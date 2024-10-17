# varDFE

Current version: 0.1.0
Last updated: 2024-10-16

## Folder structure

```
.
├── README.md
├── dfe.yml # yaml file for creating the conda environment
├── src # contents of the varDFE package
│   └── varDFE
│       ├── DFE
│       │   ├── Cache1D_mod2.py
│       │   ├── Cache1D_util.py
│       │   ├── DFEGridsearchWorker.py
│       │   ├── DFEInferenceWorker.py
│       │   ├── DemogSelModels2.py
│       │   ├── InputDFE.py
│       │   ├── OutputDFE.py
│       │   ├── PDFValidation.py
│       │   ├── PDFs2.py
│       │   └── __init__.py
│       ├── Demography
│       │   ├── DemogValidation.py
│       │   ├── Demographics1D2.py
│       │   ├── InputDemog.py
│       │   ├── OutputDemog.py
│       │   └── __init__.py
│       ├── Misc
│       │   ├── LoggerDFE.py
│       │   ├── Plotting.py
│       │   ├── Util.py
│       │   └── __init__.py
│       └── __init__.py
├── workflow # python API for running the workflows
│   ├── DFE
│   │   ├── DFE1D_gridsearch.py
│   │   ├── DFE1D_inferenceFIM.py
|   │   └── DFE1D_refspectra.py
|   └── Demography
|       └── Demog1D_sizechangeFIM.py
├── example # example folder for running the workflows
│   └── [Upcoming information]
├── pyproject.toml
└── setup.cfg
```

## Features

1. provide quick and easy demographic inference workflow for simple size change models.
2. pre-compute the DFE spectra given the inferred demographic model.
3. perform quick and parallel DFE inference for various DFE functional forms.
4. compare DFE inferred in different populations using a gridsearch approach.

## Installation

We recommend using `varDFE` in a conda environment as an editable package.

1. install [miniconda3](https://docs.anaconda.com/miniconda/miniconda-install/)
2. set up the conda environment `dfe` and install the required packages listed in `dfe.yml`
3. install `varDFE` as an editable package

```bash
# Please ensure that miniconda3 is installed before executing the following commands

# download varDFE
wget https://github.com/meixilin/varDFE/archive/refs/heads/master.zip

# unzip varDFE into your working directory
cd <your-dir>
unzip master.zip
mv varDFE-master varDFE
cd varDFE
# yaml file for creating the conda environment should be available
ls dfe.yml

# create the conda environment `dfe`
conda env create -f dfe.yml
conda activate dfe
which pip
# should be: <your-dir>/miniconda3/envs/dfe/bin/pip
pip install -e ./
```

## Documentation and tutorials

Tutorial on inferring the DFE from a Site-Frequency Spectrum (SFS) using varDFE is available in the `example` folder.

## License

varDFE is released under the GNU General Public License v3.0 license.

Disclaimer: `varDFE` and this tutorial is provided "as is" without any warranties or representations of any kind, express or implied. I make no guarantees or warranties regarding the accuracy, reliability, completeness, suitability, or timeliness of the software.

## Citation

If you use varDFE in your research, please cite our paper:

[Upcoming information]

Remember to cite the `dadi` package and `fitdadi` this package is based on as well.

```
RN Gutenkunst, RD Hernandez, SH Williamson, CD Bustamante "Inferring the joint demographic history of multiple populations from multidimensional SNP data" PLoS Genetics 5:e1000695 (2009).

BY Kim, CD Huber, KE Lohmueller "Inference of the Distribution of Selection Coefficients for New Nonsynonymous Mutations Using Large Samples" Genetics 206:345 (2017).
```

## Contact

For questions and support, please open an [issue](https://github.com/meixilin/varDFE/issues) on GitHub.
