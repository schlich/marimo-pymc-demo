import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", layout_file="layouts/demo.slides.json")

with app.setup:
    import pymc as pm
    import numpy as np
    import arviz as az
    import arviz_plots as azp


@app.cell
def _(mo):
    mo.md(
        r"""
    # Tightening the Bayesian Feedback Loop with Marimo

    ### Ty Schlichenmeyer

    ---
    """
    )

    mo.hstack(
        [
            mo.md("""
            ## About me & my Bayesian journey:

            - PhD in Neuroengineering @ Washington University in St Louis
            - Data Engineer in industry since 2021 (Leasing, Marketing)
            - Since April, contract work on Bayesian pricing model for aircraft resale value with *Aerotrends, ltd.*
            - Looking for new opportunities!
            """),
            mo.image(
                "inspectFit.jpg",
                caption="Figure from Palamedes Toolbox (MATLAB) documentation. Bayesian logistic regression is a standard method at the core of 'psychophysics' - the science of quantifying sensation",
            ),
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
            - Chained execution allows model builders to be more exploratory in initial phases and build a deep intutition for details of Bayesian mechanisms and libraries.
            """),
        ],
        widths="equal",
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    prior = (
        mo.md("""
        **Prior hyperparameters**

        {mu}

        {sigma}
    """).batch(
            mu=mo.ui.number(value=0, label=r"$\mu_{height} (cm)$"),
            sigma=mo.ui.number(value=10, label=r"$\sigma_{height} (cm)$"),
        )
        # .form(show_clear_button=True, bordered=False)
    )
    return (prior,)


@app.cell
def _(mo, prior):
    mo.md(
        r"""#### Simple example of adjusting priors: from [Exploratory Analysis of Bayesian models](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html)"""
    )

    y = np.random.normal(174, 6, 127)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=prior.value["mu"], sigma=prior.value["sigma"])
        sigma = pm.HalfNormal("sigma", sigma=10)

        y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)

        dt = pm.sample_prior_predictive(samples=100)

    fig = az.plot_ppc(dt, group="prior", kind="cumulative")
    mo.hstack(
        [
            mo.vstack(
                [
                    prior,
                    mo.md("""
        Notes: 

        - Tallest person in the world ~ 250 cm
        - Shortest person in the world ~ 50 cm
    """),
                ],
                justify="center",
            ),
            fig,
        ]
    )
    return (model,)


@app.cell(hide_code=True)
def _(sys):
    from mofresh import HTMLRefreshWidget
    import math


    class PyMCProgress:
        """
        Displays PyMC progress using mofresh's HTMLRefreshWidget.
        Updates an HTML table with per-chain and overall progress.
        """

        def __init__(
            self,
            num_chains: int,
            tune_steps_per_chain: int,
            draw_steps_per_chain: int,
        ):
            """
            Initializes the mofresh-based progress tracker.

            Args:
                num_chains (int): The total number of chains being run.
                tune_steps_per_chain (int): Number of tuning steps for each chain.
                draw_steps_per_chain (int): Number of drawing (sampling) steps for each chain.
            """
            self.num_chains = num_chains
            self.tune_steps_per_chain = tune_steps_per_chain
            self.draw_steps_per_chain = draw_steps_per_chain

            self.total_overall_iterations = (
                tune_steps_per_chain + draw_steps_per_chain
            ) * num_chains
            self.total_callbacks_received = 0

            self.chains_status = []
            for i in range(num_chains):
                self.chains_status.append(
                    {
                        "id": i,
                        "stage": "Tuning",
                        "current_steps_in_stage": 0,
                        "total_steps_in_stage": self.tune_steps_per_chain,
                        "completed_tuning": False,
                        "completed_sampling": False,
                        "divergences_sampling": 0,
                        "leapfrog_steps": 0,
                    }
                )

            # Create the widget that will display the HTML
            self.display_widget = HTMLRefreshWidget()
            self._initial_html_set = True

        def _generate_html_progress(self, is_final: bool = False) -> str:
            """
            Generates an HTML string representing the current progress.
            """
            html_parts = [
                '<table border="1" style="width:100%; border-collapse: collapse; font-family: monospace;">'
            ]
            html_parts.append(
                "<tr><th>Chain</th><th>Stage</th><th>Progress</th><th>%</th><th>Div(S)</th><th>LF Steps</th></tr>"
            )

            total_divergences_all_chains_sampling = 0

            for chain_stat in self.chains_status:
                stage = chain_stat["stage"]
                current = chain_stat["current_steps_in_stage"]
                total_stage = chain_stat["total_steps_in_stage"]
                divergences_sampling = chain_stat["divergences_sampling"]
                leapfrogs = chain_stat["leapfrog_steps"]

                total_divergences_all_chains_sampling += divergences_sampling

                if (
                    stage == "Done"
                ):  # Ensure 'Done' shows full progress for that stage
                    if (
                        chain_stat["completed_tuning"]
                        and not chain_stat["completed_sampling"]
                    ):
                        current = self.draw_steps_per_chain
                        total_stage = self.draw_steps_per_chain
                    elif not chain_stat["completed_tuning"]:
                        current = self.tune_steps_per_chain
                        total_stage = self.tune_steps_per_chain
                    else:
                        current = total_stage

                progress_frac = 0.0
                if total_stage > 0:
                    progress_frac = min(1.0, max(0.0, current / total_stage))
                elif current > 0:
                    progress_frac = 1.0

                percent_complete_stage = progress_frac * 100

                # Simple bar (optional, can be removed or enhanced with CSS)
                bar_width_chars = 20  # Number of characters for the text bar
                filled_chars = math.floor(bar_width_chars * progress_frac)
                bar_color = "grey"  # Default or for 'Done' if not fully sampled
                if stage == "Tuning":
                    bar_color = "gold"
                elif stage == "Sampling":
                    bar_color = "mediumseagreen"
                elif (
                    stage == "Done" and current >= total_stage
                ):  # Ensure 'Done' is fully green if completed
                    bar_color = "mediumseagreen"

                filled_bar_html = (
                    f'<span style="color:{bar_color};">{"█" * filled_chars}</span>'
                )
                empty_bar_html = f'<span style="color:#e0e0e0;">{"─" * (bar_width_chars - filled_chars)}</span>'
                bar_str_html = filled_bar_html + empty_bar_html

                html_parts.append(
                    f"<tr>"
                    f"<td style='text-align:center;'>C{chain_stat['id'] + 1}</td>"
                    f"<td style='text-align:center;'>{stage}</td>"
                    f"<td style='text-align:left; padding-left: 5px;'>{bar_str_html} {current}/{total_stage}</td>"
                    f"<td style='text-align:right; padding-right: 5px;'>{percent_complete_stage:.1f}%</td>"
                    f"<td style='text-align:center;'>{divergences_sampling}</td>"
                    f"<td style='text-align:center;'>{leapfrogs}</td>"
                    f"</tr>"
                )

            html_parts.append("</table>")

            # Overall Progress
            if self.total_overall_iterations > 0:
                overall_frac = 0.0
                if self.total_overall_iterations > 0:
                    overall_frac = min(
                        1.0,
                        self.total_callbacks_received
                        / self.total_overall_iterations,
                    )
                overall_percent = overall_frac * 100

                overall_summary_text = f"Overall Progress: {self.total_callbacks_received}/{self.total_overall_iterations} ({overall_percent:.1f}%)"
                if is_final:
                    if (
                        self.total_callbacks_received
                        >= self.total_overall_iterations
                    ):
                        overall_summary_text = f"Overall: Complete ({overall_percent:.1f}%). Total Sampling Divergences: {total_divergences_all_chains_sampling}"
                    else:
                        overall_summary_text = f"Overall: {self.total_callbacks_received}/{self.total_overall_iterations} ({overall_percent:.1f}%) Ended. Total Sampling Divergences: {total_divergences_all_chains_sampling}"

                html_parts.append(
                    f"<p style='font-family: monospace; margin-top: 5px;'>{overall_summary_text}</p>"
                )

            return "".join(html_parts)

        def _update_display(self, is_final: bool = False):
            """Generates HTML and updates the widget."""
            current_html = self._generate_html_progress(is_final=is_final)
            self.display_widget.html = current_html
            self._initial_html_set = True

        def callback(self, trace, draw) -> None:
            """
            Callback function to be invoked by PyMC at each step.
            """
            self.total_callbacks_received += 1
            chain_idx = draw.chain

            if not (0 <= chain_idx < self.num_chains):
                # Handle error, perhaps log it. For now, skip update.
                print(
                    f"Warning: Invalid chain index {chain_idx} received in MofreshPymcProgress.",
                    file=sys.stderr,
                )
                return

            chain_stat = self.chains_status[chain_idx]

            is_tuning_sample = False
            diverged_this_step = False
            leapfrog_steps_this_step = 0

            stats_to_check = (
                draw.stats[0]
                if isinstance(draw.stats, list) and draw.stats
                else draw.stats
            )

            if isinstance(stats_to_check, dict):
                is_tuning_sample = stats_to_check.get("tune", False)
                if stats_to_check.get("diverging", False) or stats_to_check.get(
                    "divergence", False
                ):
                    diverged_this_step = True
                leapfrog_steps_this_step = stats_to_check.get("n_steps", 0)

            if diverged_this_step and not is_tuning_sample:
                chain_stat["divergences_sampling"] += 1
            chain_stat["leapfrog_steps"] += leapfrog_steps_this_step

            if not chain_stat["completed_tuning"]:
                if (
                    not is_tuning_sample
                    and chain_stat["current_steps_in_stage"]
                    < self.tune_steps_per_chain
                ):
                    chain_stat["current_steps_in_stage"] = (
                        self.tune_steps_per_chain
                    )

                if is_tuning_sample:
                    chain_stat["stage"] = "Tuning"
                    chain_stat["total_steps_in_stage"] = self.tune_steps_per_chain
                    chain_stat["current_steps_in_stage"] += 1

                if (
                    chain_stat["current_steps_in_stage"]
                    >= self.tune_steps_per_chain
                ):
                    chain_stat["completed_tuning"] = True
                    chain_stat["stage"] = "Sampling"
                    chain_stat["current_steps_in_stage"] = 0
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                    if is_tuning_sample:
                        self._update_display()
                        return

            if (
                chain_stat["completed_tuning"]
                and not chain_stat["completed_sampling"]
            ):
                if is_tuning_sample:
                    chain_stat["stage"] = "Sampling"
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                else:
                    chain_stat["stage"] = "Sampling"
                    chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain
                    chain_stat["current_steps_in_stage"] += 1

                if (
                    chain_stat["current_steps_in_stage"]
                    >= self.draw_steps_per_chain
                ):
                    chain_stat["completed_sampling"] = True
                    chain_stat["stage"] = "Done"

            elif chain_stat["completed_sampling"]:
                chain_stat["stage"] = "Done"
                chain_stat["current_steps_in_stage"] = self.draw_steps_per_chain
                chain_stat["total_steps_in_stage"] = self.draw_steps_per_chain

            self._update_display()

        def finalize(self) -> None:
            """
            Updates the display with the final progress state.
            """
            if not self._initial_html_set and self.total_overall_iterations == 0:
                self.display_widget.html = "<p style='font-family: monospace;'>No iterations performed.</p>"
                return
            if not self._initial_html_set and self.total_overall_iterations > 0:
                # If finalize is called before any callback (e.g. error before sampling starts)
                self._update_display(is_final=False)  # Show initial 0% state

            for chain_stat in self.chains_status:
                if chain_stat["completed_sampling"]:
                    chain_stat["stage"] = "Done"
                elif chain_stat["completed_tuning"]:
                    chain_stat["stage"] = (
                        "Sampling"  # Could be 'Sampling (Ended)' if you prefer
                    )
                else:
                    chain_stat["stage"] = "Tuning"  # Could be 'Tuning (Ended)'

            self._update_display(is_final=True)

        def reset(self):
            """Resets the internal state of the progress tracker for a new run."""
            self.total_overall_iterations = (
                self.tune_steps_per_chain + self.draw_steps_per_chain
            ) * self.num_chains
            self.total_callbacks_received = 0
            self.chains_status = []
            for i in range(self.num_chains):
                self.chains_status.append(
                    {
                        "id": i,
                        "stage": "Tuning",
                        "current_steps_in_stage": 0,
                        "total_steps_in_stage": self.tune_steps_per_chain,
                        "completed_tuning": False,
                        "completed_sampling": False,
                        "divergences_sampling": 0,
                        "leapfrog_steps": 0,
                    }
                )
            self._initial_html_set = (
                False  # Ensure the display updates from scratch
            )

        def __enter__(self):
            """
            Enter the runtime context. Sets initial HTML.
            """
            self.reset()  # Reset state at the beginning of each 'with' block
            self._update_display()  # Update display to show initial (reset) state
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            Exit the runtime context. Ensures finalize() is called.
            """
            self.finalize()
            return False  # Re-raise any exceptions
    return (PyMCProgress,)


@app.cell
def _(PyMCProgress):
    progress_tracker = PyMCProgress(
        num_chains=4,
        tune_steps_per_chain=200,
        draw_steps_per_chain=200,
    )
    return (progress_tracker,)


@app.cell
def _(mo, progress_tracker):
    progress_tracker
    run_button = mo.ui.run_button(label="Run Inferencing!")
    run_button
    return (run_button,)


@app.cell
def _(mo, progress_tracker):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        "## A custom Anywidget progress tracker for the MC Sampler"
                    ),
                    mo.md("---"),
                    mo.md(
                        "View this notebook on **molab** for source code & example details."
                    ),
                    mo.md(
                        "> Note: There is an open [pull request](https://github.com/pymc-devs/pymc/pull/7883) in the PyMC repo for out-of-the-box progress tracking!"
                    ),
                    progress_tracker.display_widget,
                ]
            ),
            mo.md("""
        ```python
        with progress_tracker:
            with model:
                idata = pm.sample(
                    tune=200,
                    draws=200,
                    chains=4,
                    callback=progress_tracker.callback,  
                    # ^^^ this is where you put the callback
                )
        ```
        """),
        ]
    )
    return


@app.cell
def _(model, progress_tracker, run_button):
    with progress_tracker:
        if run_button.value:
            with model:
                idata = pm.sample(
                    tune=200,
                    draws=200,
                    chains=4,
                    callback=progress_tracker.callback,  # <-- this is where you put the callback
                )
    return (idata,)


@app.cell
def _(mo):
    ll_button = mo.ui.run_button(
        label="Calculate log likelihood & posterior predictive checks"
    )
    return (ll_button,)


@app.cell
def _(idata, idata_w_ppc, mo):
    mo.hstack(
        [
            mo.md(r"""
    ### Stateful **InferenceData** xarray-based object is de-facto in memory storage object for model fitting results (Arviz, PyMC)

    - Fracturing of data types/objects between PyMC, Arviz, and Bambi cause friction.
    - Need to manage memory, disk, computation time

    ```python
    with model:
        idata_w_ll = pm.compute_log_likelihood(idata)
        idata_w_ppc = pm.sample_posterior_predictive(idata)

    idata_w_ppc
    ```
    """),
            mo.hstack(
                [
                    idata,
                    idata_w_ppc,
                ],
                justify="start",
            ),
        ],
        widths="equal",
    )
    return


@app.cell
def _(ll_button):
    ll_button
    return


@app.cell
def _(idata, ll_button, model):
    if ll_button.value:
        with model:
            idata_w_ll = pm.compute_log_likelihood(idata)
            idata_w_ppc = pm.sample_posterior_predictive(idata)
    else:
        idata_w_ppc = None
    return (idata_w_ppc,)


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md(
                r"""
    ### Utilize marimo caching!

    - Data retrieval and updating
    - Avoiding recalculation of expensive steps of model parameters (e.g. prior hyperparameters, n sampling steps) we've **already tried**.

    """
            ),
            mo.md("""

    ```python
    import marimo as mo
    from pydantic import BaseModel

    class SamplerConfig(BaseModel):
        tuning_samples: int
        posterior_samples: int

    with mo.persistent_cache("prior_predctive"):
        with model:
            idata_cached = pm.sample_prior_predictive()

    idata_cached
    ```
    """),
        ]
    )
    return


@app.cell(disabled=True)
def _(mo, model):
    # @mo.persistent_cache
    # def run_sampling(model: pm.Model) -> az.InferenceData:

    with mo.persistent_cache("prior_predctive"):
        with model:
            idata_cached = pm.sample_prior_predictive()
    idata_cached
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What's next?

    **Frequent OOM errors and long convergence times during model training point to need for a more "online" model tuning process and thoughtful data management (esp for large datasets**).**

    - Utilize PyMC callback function for more immediate feedback and clearer intution of diagnostic statistics
    - Utilize and incorporate Arviz preview features and ZarrTrace sampling backend/storage
    - Translate vast canonical example notebook library from PyMC/Bambi/Arviz
    - Investigate more control flow mechanisms

    ---
    ## Thank you!

    > ty.schlich@gmail.com
    >
    > Ping me on Discord! **schlich** / *ty.schlich*
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
