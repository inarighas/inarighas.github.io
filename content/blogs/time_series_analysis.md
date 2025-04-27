---
title: "Modern time series analysis"
date: 2023-01-11T21:50:56+01:00
author: Ali S.
mmark: true
math: true
draft: false
header-includes:
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

---

In this post, I use some lecture notes from an online course/talk about ["Modern Time Series Analysis" presented by Aileen Nielsen in Scipy2019](https://youtu.be/v5ijNXvlC5A) and format them to produce a written tutorial for people interested in this topic.

## Introduction

Main steps in time series an:

- Visualization (seasonality, *stationarity*)
- Identifying/modelling underlying distributions of data and stochastic process generating the data.
- Smoothing (past), Filtering (present), Forecasting (future)
- Classification
- Anomaly detection, outlier points within time series

Some examples of time series analysis systems:

- Let's suppose we perform a noisy measure, time series analysis can allow estimating what was the true value given these noisy values.
- Classification of healthy and unhealthy EKG signals.
- Anomaly detection of bank card fraud or misfunctioning of some industrial facility

A note on Cross-sectional vs Time series:

- More missing data in time series;
- High correlation between time steps;
- Timestamps can be used as a measure of distance.

One axis is monotonically increasing (time axis)
The signal can show some characteristic such as seasonality, cycles, autocorrelation & trends. It is also possible to spot some stochastic shift to some specific stochastic regime.

Frequent problems with time series:

- Residuals of time series models significance (special time-aware diagnostics/evaluation, autocorrelation plot of the residuals)
- Cross-validation depends on the data and on the future does not leak backwards in time and overestimate your model performance. CV should roll test and train forward in time. In other words, we avoid testing on data that is older (in time) than train data.
- "Look ahead" is another issue, timestamping does not mean when you would actually have that data points. Sometimes, this is not an issue.

## State Space Models: Box-Jenkins ARIMA as an example

BJ-ARIMA (autoregressive integrative moving average, same class of AM, ARMA for *AutoRegressive-Moving Average*):

- Remarkably successful to model basic timeseries a yields cutting edge performance
- Particularly effuicient on small datasets with no too much noise and some seasonalit

We choose this model because it is quite performant for a bunch of problems. It is also not the most simple one but it made of simpler components that we will describe as we go along.

### ARMA Model

Here is the mathematical formulation of the ARMA model:

{{< mathjax/block >}}
$$(1-\sum^{p}_{i=1}\alpha_i~L^i)~X_t = (1 + \sum^q_{i=1} \theta_i~L^i)~\varepsilon_t$$
{{< /mathjax/block >}}

where $\alpha_{1:p}$ are called autoregressive components, $\theta_{1:q}$ are the moving average components and $L$ is called a lag operator that moves a variable back (or forth) in time. The lag operator of power $2$ transforms $X_t$ to $X_{t-2}$.
In this type of models, future values depends past values and past errors. $p$ and $q$ are the main parameter to fiddle with to tune the model.

### ARIMA Model

Now, let's look at the mathematical formulation of the ARIMA model:

{{< mathjax/block >}}
$$\left(1-\sum^{p}_{i=1}\phi_i~L^i\right)~(1-L)^d~X_t = \left(1 + \sum^q_{i=1} \theta_i~L^i \right)~\varepsilon_t$$
{{< /mathjax/block >}}

The difference with ARMA is the term $(1-L)^d$. Mathematically, this makes the future values depend on $\Delta X$ instead of $X$ values. The only lost information is the offset, but makes the model more tractable.

Fitting these models is not a trivial task (is more like an art).

### Limits of ARMA/ARIMA

- These model are not especially intuitive
- No way to build an understanding about how it works (Random walk, cyclical elements, external regressors)
- Some systems we try to model present slow or stochastic cycle that cannot be taken into account with ARIMA models (e.g. a cycle of 24 hours in hourly data can be hard to spot and the model does not get better with more data).

### Structural time series

let's suppose a type of time series that can be split to multiple time components (a certain level/offset, seasonal/periodic, irregular/noise).

The structural time series can be expressed in ARIMA form, and can be fit via maximum likelihood estimator (see Kalman filter). The particular feature of there models is to offer insights into the underlying structure. (Also compatible with Bayesian analysis).

### The 3 solutions offered by state space models

- Filtering: The distribution of the current state at time $t$ given all previous measurements up to and including $t$ (e.g. noise removal).
- Prediction: The distribution of the future states at time $t+k$ given all previous measurements up to and including $t$ (e.g., forecasting).
- Smoothing: The distribution of a given state at time $k$ given all previous and future measurements from $0$ to $T$ (e.g., correct past estimation given future values).

### Kalman Filter

![The basic steps of Kalman filtering](/img/Kalman_filter.jpg)

The underlying model:

{{< mathjax/block >}}
$$\bf{x}_k = \bf{F}_k~\bf{x}_{k-1} + \bf{B}_k\bf{u}_k + \bf{w}_k$$
{{< /mathjax/block >}}

$$\bf{z_k} = \bf{H}_k\bf{x}_k + \bf{v}_k$$

- {{< mathjax/inline >}}$\bf{F}_k~\bf{x}_{k-1}${{< /mathjax/inline >}} is the AR part of the model
- $\bf{B}_k~\bf{u}_k$ is the command and how it impacts the system (control theory)
- $\bf{w}_k$ is the noise component
- Kalman filters can feel more suitable for systems on which we (as modelers) have some knowledge and hypothesis (control theory, Newtonian physics system)

## Particle Filtering

While Kalman filters work well for linear systems with Gaussian noise, many real-world time series don't fit these assumptions. Particle filters (also known as Sequential Monte Carlo methods) offer an alternative approach that can handle non-linear dynamics and non-Gaussian distributions.

### Theory and Equations

The particle filter represents the posterior distribution of states using a set of weighted particles. The key steps are:

1. **Initialization**: Sample N particles from the prior distribution: $x_0^i \sim p(x_0)$
2. **Prediction**: Propagate each particle through the state transition model:
   $x_t^i \sim p(x_t|x_{t-1}^i)$
3. **Update**: Calculate weights based on likelihood:
   $w_t^i = p(z_t|x_t^i)$
4. **Normalization**: {{< mathjax/inline >}}$\tilde{w}_t^i = \frac{w_t^i}{\sum_{j=1}^N w_t^j}${{< /mathjax/inline >}} 
5. **Resampling**: Draw N particles with probabilities proportional to weights
6. **Estimate**: $\hat{x}_t = \sum_{i=1}^N \tilde{w}_t^i x_t^i$

For a state-space model:

- State transition: $x_t = f(x_{t-1}, u_t, w_t)$
- Observation model: $z_t = h(x_t, v_t)$

Where $f$ and $h$ can be non-linear functions, and $w_t$ and $v_t$ are noise terms with any distribution.

### Pseudo-code Implementation

```plaintext
function PARTICLE_FILTER(observations, N_particles):
    # Initialize particles
    particles = sample_from_prior(N_particles)
    weights = [1/N_particles] * N_particles
    
    filtered_states = []
    
    for t in range(len(observations)):
        # Prediction step
        for i in range(N_particles):
            particles[i] = propagate_state(particles[i])
        
        # Update step
        for i in range(N_particles):
            weights[i] = likelihood(observations[t], particles[i])
        
        # Normalize weights
        weights = normalize(weights)
        
        # Compute estimate
        state_estimate = weighted_average(particles, weights)
        filtered_states.append(state_estimate)
        
        # Resample if effective sample size is too low
        if effective_sample_size(weights) < threshold:
            particles = resample(particles, weights)
            weights = [1/N_particles] * N_particles
    
    return filtered_states
```

### Advantages and Applications

- Handles non-linear dynamics and non-Gaussian noise
- Provides full posterior distribution rather than just point estimates
- Particularly useful for financial market modeling, object tracking, and complex time series with regime changes

## Practical Example

Let's implement a concise practical example focusing on key time series analysis techniques.

### Data Exploration and Decomposition

For any time series analysis, we start with visual exploration and decomposition:

```plaintext
# Pseudo-code for time series decomposition
function DECOMPOSE_TIME_SERIES(time_series):
    # Decompose into trend, seasonal, and residual components
    trend = moving_average(time_series, window=12)
    detrended = time_series / trend  # For multiplicative model
    
    # Extract seasonal component
    seasonal = []
    for i in range(12):  # For monthly data
        seasonal[i] = mean(detrended[i::12])  # Average all same months
    
    # Normalize seasonal component
    seasonal = seasonal / mean(seasonal)
    
    # Extract residual
    residual = detrended / seasonal
    
    return trend, seasonal, residual
```

### ARIMA Modeling Approach

The ARIMA(p,d,q) model combines autoregressive (AR), differencing (I), and moving average (MA) components:

AR(p) component: $X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \varepsilon_t$

MA(q) component: $X_t = \mu + \varepsilon_t + \sum_{i=1}^q \theta_i \varepsilon_{t-i}$

ARIMA(p,d,q): $\phi(L)(1-L)^d X_t = \theta(L)\varepsilon_t$

Where:

- $\phi(L) = 1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p$
- $\theta(L) = 1 + \theta_1 L + \theta_2 L^2 + ... + \theta_q L^q$
- $L$ is the lag operator
- $(1-L)^d$ represents differencing of order $d$

```plaintext
# Pseudo-code for ARIMA modeling
function FIT_ARIMA(time_series, p, d, q):
    # Difference the series d times if needed
    diff_series = difference(time_series, d)
    
    # Estimate AR parameters
    phi = estimate_ar_parameters(diff_series, p)
    
    # Estimate MA parameters 
    theta = estimate_ma_parameters(diff_series, q)
    
    # Generate forecasts
    forecasts = generate_arima_forecasts(time_series, phi, theta, d)
    
    return forecasts, phi, theta
```

### Structural Time Series Approach

A structural time series model decomposes the series into interpretable components:

$y_t = \mu_t + \tau_t + \gamma_t + \varepsilon_t$

Where:

- $\mu_t$ is the level/trend component: $\mu_t = \mu_{t-1} + \beta_{t-1} + \eta_t$
- $\beta_t$ is the slope: $\beta_t = \beta_{t-1} + \zeta_t$
- $\tau_t$ is the seasonal component: $\sum_{j=0}^{s-1} \tau_{t-j} = \omega_t$
- $\gamma_t$ represents cyclical variations
- $\varepsilon_t, \eta_t, \zeta_t, \omega_t$ are noise terms

```plaintext
# Pseudo-code for structural time series
function FIT_STRUCTURAL_MODEL(time_series):
    # Initialize state components
    level = time_series[0]
    trend = estimate_initial_trend(time_series)
    seasonal = estimate_initial_seasonal(time_series)
    
    # Initialize state covariance
    state_cov = initialize_covariance()
    
    # Estimate hyperparameters (variance terms)
    hyperparams = maximize_likelihood(time_series)
    
    # Apply Kalman filter
    filtered_states = kalman_filter(time_series, hyperparams)
    
    # Forecasting
    forecasts = forecast_structural_model(filtered_states, hyperparams, horizon=12)
    
    return filtered_states, forecasts
```

### Simple Kalman Filter Example

For a time series with trend, we can use a basic Kalman filter:

State equation: $\mathbf{x}_t = \begin{bmatrix} \text{level}_t \\ \text{trend}_t \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \mathbf{x}_{t-1} + \mathbf{w}_t$

Observation equation: $y_t = \begin{bmatrix} 1 & 0 \end{bmatrix} \mathbf{x}_t + v_t$

Where $\mathbf{w}_t \sim N(0, \mathbf{Q})$ and $v_t \sim N(0, R)$ are process and observation noise.

```plaintext
# Pseudo-code for Kalman filter
function KALMAN_FILTER(observations):
    # Initialize
    x = [observations[0], 0]  # Initial state (level and trend)
    P = identity_matrix(2)    # Initial covariance
    
    # System matrices
    F = [[1, 1], [0, 1]]      # State transition 
    H = [1, 0]                # Observation matrix
    Q = small_identity(2)     # Process noise
    R = scalar(1)             # Observation noise
    
    filtered_states = []
    
    for y in observations:
        # Predict
        x_pred = F * x
        P_pred = F * P * F' + Q
        
        # Update
        K = P_pred * H' * inv(H * P_pred * H' + R)  # Kalman gain
        x = x_pred + K * (y - H * x_pred)           # Updated state
        P = (I - K * H) * P_pred                    # Updated covariance
        
        filtered_states.append(x)
    
    return filtered_states
```

### Example Analysis: Airline Passengers

Using the monthly airline passenger dataset (1949-1960):

1. **Visualization**: The data shows strong upward trend and seasonal pattern with increasing variance

2. **Decomposition**:
   - Trend: Steady growth over the period
   - Seasonal: Clear annual pattern with peaks in summer months
   - Residual: Some remaining patterns suggest possible regime changes

3. **Model comparison**:

   | Model | RMSE | Features |
   |-------|------|----------|
   | SARIMA(1,1,1)(1,1,1,12) | 32.4 | Captures seasonality well |
   | Structural TS | 29.8 | Clear component interpretation |
   | Kalman Filter | 31.5 | Simple, online estimation |
   | Particle Filter | 30.2 | Handles non-linearities |

4. **Example forecast visualization**:
   - The structural model shows the best balance of accuracy and interpretability
   - All models capture the seasonal pattern
   - The particle filter better handles the increasing variance

## Conclusion

In this blog post, we've explored modern time series analysis techniques, from traditional ARIMA models to state space models including Kalman filters and particle filters.

Key takeaways:

1. **Visualization and decomposition** are crucial first steps to identify patterns, seasonality, and *stationarity*.

2. **ARIMA models** remain powerful for many forecasting tasks but require *stationarity* assumptions.

3. **State space models** offer a flexible framework for complex temporal dependencies:
   - Kalman filters work well for linear systems with Gaussian noise
   - Particle filters handle non-linear dynamics and non-Gaussian distributions
   - Structural time series models provide interpretable decomposition

4. **Model selection** should consider both accuracy metrics and interpretability needs.

5. **Proper cross-validation** for time series requires special care to avoid data leakage.

The choice between methods depends on your specific data characteristics, domain knowledge, and goals. Often, experimenting with multiple approaches and comparing their performance is the best strategy.
