# Advanced Econometrics (Econometrics II)

![Course header image](src/images/ecm-2-header-img.png)

## Short Description

[![DOI](https://zenodo.org/badge/947354285.svg)](https://doi.org/10.5281/zenodo.15463535)
 

This repository contains the lecture materials and supporting content for the undergraduate course *Advanced Econometrics (Econometrics II)* that I offered at the University of Bonn.

The course builds on the basic econometrics course in three directions:

1. Considering both causal inference and forecasting.
2. Introducing further empirical methods.
3. Discussing the underlying theory.

The course is designed for students with prior exposure to basic statistics and econometrics. The empirical illustrations are implemented in Python.

## Course Overview

**Instructor:** Vladislav Morozov 

**Course Website:** <https://vladislav-morozov.github.io/econometrics-2/>

**Level:** Undergraduate

**Prerequisites:** Prior coursework in statistics and econometrics
 

**Applications in:** [![Python](https://img.shields.io/badge/python-ffdd54?style=flat&logo=python&logoColor=3776AB)](https://www.python.org/)




The course is structured as follows: 

1. A deeper look at linear regression:
   - A vector-matrix form approach to linear regression.
   - Basics of identification analysis. 
   - Asymptotic theory for the OLS estimator.
2. Asymptotic inference:
   - Refresher: key definitions and intuition of hypothesis testing.
   - Tests for linear hypotheses: $t$- and Wald tests.
   - The delta method and nonlinear Wald tests.

3. Panel data in causal settings:
   - Event studies.
   - Differences-in-differences.
   - Two-way fixed effect approaches with multivalued treatment.
   - Mean group estimation.
4. Introduction to statistical and machine learning:
   - Causal inference vs. forecasting. 
   - Components of a machine learning problem.
   - PAC learnability.
   - The no-free-lunch theorem.
   - Regression and classification problems in practice.
 

## Course Materials

The course draws from a range of textbooks. Relevant chapters are indicated within the lecture slides.

* **Brockwell & Davis** (2016). *Introduction to Time Series and Forecasting*. Springer.
* **Cunningham** (2021). *Causal Inference: The Mixtape*. Yale University Press.
* **Huntington-Klein** (2025). *The Effect: An Introduction to Research Design and Causality*. CRC Press.
* **James et al.** (2023). *An Introduction to Statistical Learning (Python Edition)*. Springer.
* **Wooldridge** (2020). *Introductory Econometrics* (7th ed.). Cengage.

All materials are available either online or through the university library network.

## Assessment

Final evaluation is based on a 90-minute written, closed-book exam. The course offers ungraded problem sets.

## About This Repository

This repository serves as the central source for lecture slides and problem sets for the course. The materials are made with ![Quarto](https://img.shields.io/badge/Quarto-39729E?style=flat&logo=quarto&logoColor=white).

Contributions, typo reports, and suggestions for clarity are welcome. Please open an issue or contact me directly!

## Building the Presentations

To build the website and all the slides, execute the following steps.

- Repo preparation:
    - Clone the repository.
    - Download the large excluded `country-quarter.tab` dataset from the [Harvard Dataverse](https://dataverse.harvard.edu/file.xhtml?fileId=6425134&version=1.0) and place it under `src/slides/panel/data`. 
- Install Quarto from [https://quarto.org/docs/get-started/](https://quarto.org/docs/get-started/).
- Change to `src`, set up the virtual environment with `uv sync`, and install the required Quarto extensions:
    ```bash
    quarto add pandoc-ext/diagram \
    && quarto add coatless-quarto/embedio \
    && quarto add shafayetShafee/reveal-header
    ```
- Build the website and slides with `quarto render`.

This project uses the following Quarto extensions:

- [diagram](https://github.com/pandoc-ext/diagram)
- [embedio](https://github.com/coatless-quarto/embedio)
- [reveal-header](https://github.com/shafayetShafee/reveal-header)

## License

Course materials provided here are for educational use only and are licensed under the MIT license. 
