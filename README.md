# Project Title

Benford MGT

## Description

To evaluate the cross-lingual robustness of Benford-law-based machine-generated text detection, we design a multilingual experimental framework using Slovene, Czech, Slovak, and English texts from the \textit{MultiSocial} benchmark. The experiments are structured progressively, beginning with the validation of the core statistical hypothesis and subsequently extending toward supervised and cross-lingual detection settings.


## Getting Started

### Dependencies

* torch
* transformers
* pandas
* numpy
* scipy
* scikit-learn
* matplotlib
* tqdm

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Jernej Vičič

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments





Input:
    language, text, label

Output:
    language, label, model_name,
    kl, chi2, mse, r2,
    n_tokens, n_numbers
