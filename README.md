# Advanced Econometrics (Econometrics II)

This repository contains the lecture materials and supporting content for the undergraduate course *Advanced Econometrics (Econometrics II)* that I offered at the University of Bonn.

The course builds on the basic econometrics course in three directions:

1. Considering both causal inference and forecasting.
2. Introducing further empirical methods.
3. Discussing the underlying theory.

The course is designed for students with prior exposure to basic statistics and econometrics. The empirical illustrations are implemented in Python.

## Course Information

**Instructor:** Vladislav Morozov 

**Course Website:** <https://vladislav-morozov.github.io/econometrics-2/>

**Level:** Undergraduate

**Prerequisites:** Prior coursework in statistics and econometrics

**Current course status**: In progress, these materials get updated regularly

## Course Overview

The course is structured around five major components (subject to change):

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
4. Introduction to forecasting:
   - Causal inference vs. forecasting I.
   - Notions of forecast optimality.
   - Forecasting in cross-sections.
5. Parametric nonlinear models:
   - Beyond linearity: nonlinear regression and nonlinear least squares.
   - Discrete outcomes in causal settings. 
   - Elements of asymptotic theory for nonlinear models.
   - Classification as forecasting with discrete outcomes.

If time allows, we will further discuss:

6. Generalized method of moments.
   - Linear generalized method of moments (GMM).
   - IV estimation of dynamic panel data models.
   - Fundamentals of nonlinear GMM.
7.  Time series:
   - Time series as probabilistic objects and their properties.
   - Univariate models: ARIMA(X).
   - Multivariate time series: VARIMA(X).
   - Elements of causal inference with time series. 
   - Forecasting with time series vs. forecasting with panel data

Even further topics such as quantile regression, experimentation under interference, and high-dimensional data may be introduced as time allows.

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

This repository serves as the central source for lecture slides and problem sets for the course. Overall, the course is deployed as a Quarto website. 

Contributions, typo reports, and suggestions for clarity are welcome. Please open an issue or contact me directly!

## License

Course materials provided here are for educational use only and are licensed under the MIT license. 
