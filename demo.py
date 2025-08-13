import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", layout_file="layouts/demo.slides.json")

with app.setup:
    import pymc as pm
    import numpy as np
    import arviz.preview as azp


@app.cell
def _(mo):
    mo.md(
        r"""
    # Tightening the Bayesian Feedback Loop with Marimo

    ### Ty Schlichenmeyer

    ---
    """)

    mo.hstack(
        [
            mo.md("""
            ## About me & my Bayesian journey:
        
            - PhD in Neuroengineering @ Washington University in St Louis
            - Data Engineer in industry since 2021 (Leasing, Marketing)
            - Since April, contract work on Bayesian pricing model for aircraft resale value with *Aerotrends, ltd.*
            - Looking for new opportunities!
            """),
            mo.image("inspectFit.jpg", caption="Figure from Palamedes Toolbox (MATLAB) documentation. Bayesian logistic regression is a standard method at the core of 'psychophysics' - the science of quantifying sensation")
        ]
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.image("Bayesian_workflow.png"),
            mo.md("""
            - Bayesian model-building involves a complex and time consuming iteration process
            - Out of the box, Marimo's reactive execution saves countless hours of recomputation as adjustments are made at various levels
            - Chained execution allows model builders to be more exploratory in initial phases and build a deep intutition for details of bayesian mechanisms and libraries.
            """),
        ],
        widths="equal",
        align="center"
    )
    return


@app.cell
def _():
    y = np.random.normal(174, 6, 127)
    return (y,)


@app.cell
def _(mo, y):
    mo.md(r"""#### Simple example of adjusting priors: from [Exploratory Analysis of Bayesian models](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html)""")



    with pm.Model() as model: 
        # Priors for unknown model parameters
        mu = pm.Normal('mu', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)
        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
        # draw 500 samples from the prior predictive
        dt = pm.sample_prior_predictive(samples=500)

    pc = azp.plot_ppc(dt, group="prior_predictive")
    pc.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    [My deep dive - and hiccups along the way]

    ### Progress bar doesnt work out of the box in Marimo
    - Vibe-coded a custom Anywidget w/inspiration from Vincent
    - Current/recent PR for marimo compatibility
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    [My deep dive - and hiccups along the way]

    ### Stateful **InferenceData** xarray-based object is de-facto in memory storage object for model fitting results
    - Argument `inplace=True` is more "marimo-style" but comes with its own tradeoffs, especially memory usage
    -
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""### Marimo's built-in persistence features cannot store Bambi model objects""")
    return


@app.cell
def _(mo):
    mo.md(r"""But forms for model configuration parameters worked great!""")
    return


@app.cell
def _(mo):
    mo.md(r"""Frequent OOM errors and long convergence times during model training point to need for a more "online" model tuning process.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    What's next?

    - Utilize PyMC callback function for more immediate feedback and clearer intution of diagnostic statistics
    - Utilize and incorporate Arviz preview features and ZarrTrace sampling backend/storage
    - Translate vast canonical example notebook library from PyMC/Bambi/Arviz
    - Investigate more advanced control flow mechanisms
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
