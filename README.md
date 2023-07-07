# Building an ML Pipeline for Short-Term Rental Prices in NYC
We are building a pipeline for a property management company renting rooms and properties for short periods 
of time on various rental platforms. We have to estimate the typical price for a given property based on the
price of similar properties. The company received new data in bulk every week. This means that our model needs 
to be retrained with the same cadence, necessitation an end-to-end and reusable pipeline.

### Exploratory Data Analysis (EDA)
The scope of this section is to get an idea of how the process of an EDA works in the context of
pipelines, during the data exploration phase. In a real scenario, we would spend a lot more time
in this phase, but here we are going to do the bare minimum.

NOTE: remember to add some markdown cells explaining what you are about to do, so that the
notebook can be understood by other people like your colleagues

1. The ``main.py`` script already comes with the download step implemented. Run the pipeline to 
   get a sample of the data. The pipeline will also upload it to Weights & Biases:
   
  ```bash
  > mlflow run . -P steps=download
  ```
  
  You will see a message similar to:

  ```
  2021-03-12 15:44:39,840 Uploading sample.csv to Weights & Biases
  ```
  This tells us that the data is going to be stored in W&B as the artifact named ``sample.csv``.

2. Now execute the `eda` step:
   ```bash
   > mlflow run src/eda
   ```
   This will install Jupyter and all the dependencies for `pandas-profiling`, and open a Jupyter notebook instance.
   Click on New -> Python 3 and create a new notebook. Rename it `EDA` by clicking on `Untitled` at the top, beside the
   Jupyter logo.
3. Within the notebook, fetch the artifact we just created (``sample.csv``) from W&B and read 
   it with pandas:
    
    ```python
    import wandb
    import pandas as pd
    
    run = wandb.init(project="nyc_airbnb", group="eda", save_code=True)
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)
    ```
    Note that we use ``save_code=True`` in the call to ``wandb.init`` so the notebook is uploaded and versioned
    by W&B.

4. Using `pandas-profiling`, create a profile:
   ```python
   import pandas_profiling
   
   profile = pandas_profiling.ProfileReport(df)
   profile.to_widgets()
   ```
   what do you notice? Look around and see what you can find. 
   
   For example, there are missing values in a few columns and the column `last_review` is a 
   date but it is in string format. Look also at the `price` column, and note the outliers. There are some zeros and 
   some very high prices. After talking to your stakeholders, you decide to consider from a minimum of $ 10 to a 
   maximum of $ 350 per night.
   
5. Fix some of the little problems we have found in the data with the following code:
    
   ```python
   # Drop outliers
   min_price = 10
   max_price = 350
   idx = df['price'].between(min_price, max_price)
   df = df[idx].copy()
   # Convert last_review to datetime
   df['last_review'] = pd.to_datetime(df['last_review'])
   ```
   Note how we did not impute missing values. We will do that in the inference pipeline, so we will be able to handle
   missing values also in production.
6. Create a new profile or check with ``df.info()`` that all obvious problems have been solved
7. Terminate the run by running `run.finish()`
8. Save the notebook, then close it (File -> Close and Halt). In the main Jupyter notebook page, click Quit in the
   upper right to stop Jupyter. This will also terminate the mlflow run. DO NOT USE CRTL-C

## Data cleaning

Now we transfer the data processing we have done as part of the EDA to a new ``basic_cleaning`` 
step that starts from the ``sample.csv`` artifact and create a new artifact ``clean_sample.csv`` 
with the cleaned data:

1. Make sure you are in the root directory of the starter kit, then create a stub 
   for the new step. The new step should accept the parameters ``input_artifact`` 
   (the input artifact), ``output_artifact`` (the name for the output artifact), 
   ``output_type`` (the type for the output artifact), ``output_description`` 
   (a description for the output artifact), ``min_price`` (the minimum price to consider)
   and ``max_price`` (the maximum price to consider):
   
   ```bash
   > cookiecutter cookie-mlflow-step -o src
   step_name [step_name]: basic_cleaning
   script_name [run.py]: run.py
   job_type [my_step]: basic_cleaning
   short_description [My step]: A very basic data cleaning
   long_description [An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
   parameters [parameter1,parameter2]: input_artifact,output_artifact,output_type,output_description,min_price,max_price
   ```
   This will create a directory ``src/basic_cleaning`` containing the basic files required 
   for a MLflow step: ``conda.yml``, ``MLproject`` and the script (which we named ``run.py``).
   
2. Modify the ``src/basic_cleaning/run.py`` script and the ML project script by filling the 
   missing information about parameters (note the 
   comments like ``INSERT TYPE HERE`` and ``INSERT DESCRIPTION HERE``). All parameters should be
   of type ``str`` except ``min_price`` and ``max_price`` that should be ``float``.
   
3. Implement in the section marked ```# YOUR CODE HERE     #``` the steps we 
   have implemented in the notebook, including downloading the data from W&B. 
   Remember to use the ``logger`` instance already provided to print meaningful messages to screen. 
   
   Make sure to use ``args.min_price`` and ``args.max_price`` when dropping the outliers 
   (instead of  hard-coding the values like we did in the notebook).
   Save the results to a CSV file called ``clean_sample.csv`` 
   (``df.to_csv("clean_sample.csv", index=False)``)
   **_NOTE_**: Remember to use ``index=False`` when saving to CSV, otherwise the data checks in
               the next step might fail because there will be an extra ``index`` column
   
   Then upload it to W&B using:
   
   ```python
   artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
   ```
   
   **_REMEMBER__**: Whenever you are using a library (like pandas), you MUST add it as 
                    dependency in the ``conda.yml`` file. For example, here we are using pandas 
                    so we must add it to ``conda.yml`` file, including a version:
   ```yaml
   dependencies:
     - pip=20.3.3
     - pandas=1.2.3
     - pip:
         - wandb==0.10.31
   ```
   
4. Add the ``basic_cleaning`` step to the pipeline (the ``main.py`` file):

   **_WARNING:_**: please note how the path to the step is constructed: 
                   ``os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning")``.
   This is necessary because Hydra executes the script in a different directory than the root
   of the starter kit. You will have to do the same for every step you are going to add to the 
   pipeline.
   
   **_NOTE_**: Remember that when you refer to an artifact stored on W&B, you MUST specify a 
               version or a tag. For example, here the ``input_artifact`` should be 
               ``sample.csv:latest`` and NOT just ``sample.csv``. If you forget to do this, 
               you will see a message like
               ``Attempted to fetch artifact without alias (e.g. "<artifact_name>:v3" or "<artifact_name>:latest")``

   ```python
   if "basic_cleaning" in active_steps:
       _ = mlflow.run(
            os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            "main",
            parameters={
                "input_artifact": "sample.csv:latest",
                "output_artifact": "clean_sample.csv",
                "output_type": "clean_sample",
                "output_description": "Data with outliers and null values removed",
                "min_price": config['etl']['min_price'],
                "max_price": config['etl']['max_price']
            },
        )
   ```
5. Run the pipeline. If you go to W&B, you will see the new artifact type `clean_sample` and within it the 
   `clean_sample.csv` artifact

### Data splitting
Use the provided component called ``train_val_test_split`` to extract and segregate the test set. 
Add it to the pipeline then run the pipeline. As usual, use the configuration for the parameters like `test_size`,
`random_seed` and `stratify_by`. Look at the `modeling` section in the config file.

**_HINT_**: The path to the step can
be expressed as ``mlflow.run(f"{config['main']['components_repository']}/train_val_test_split", ...)``.

You can see the parameters accepted by this step [here](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/train_val_test_split/MLproject)

After you execute, you will see something like:

```
2021-03-15 01:36:44,818 Uploading trainval_data.csv dataset
2021-03-15 01:36:47,958 Uploading test_data.csv dataset
```
in the log. This tells you that the script is uploading 2 new datasets: ``trainval_data.csv`` and ``test_data.csv``.

### Train Random Forest
Complete the script ``src/train_random_forest/run.py``. All the places where you need to insert code are marked by
a `# YOUR CODE HERE` comment and are delimited by two signs like `######################################`. You can
find further instructions in the file.

Once you are done, add the step to ``main.py``. Use the name ``random_forest_export`` as ``output_artifact``.

**_NOTE_**: the main.py file already provides a variable ``rf_config`` to be passed as the
            ``rf_config`` parameter.

### Optimize hyperparameters
We Re-run the entire pipeline varying the hyperparameters of the Random Forest model. This can be
accomplished easily by exploiting the Hydra configuration system. Use the multi-run feature (adding the `-m` option 
at the end of the `hydra_options` specification), and try setting the parameter `modeling.max_tfidf_features` to 10, 15
and 30, and the `modeling.random_forest.max_features` to 0.1, 0.33, 0.5, 0.75, 1.

HINT: if you don't remember the hydra syntax, you can take inspiration from this is example, where we vary 
two other parameters (this is NOT the solution to this step):
```bash
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```
you can change this command line to accomplish your task.


### Select the best model
Go to W&B and select the best performing model. We are going to consider the Mean Absolute Error as our target metric,
so we are going to choose the model with the lowest MAE.

![wandb](images/wandb_select_best.gif "wandb")

**_HINT_**: you should switch to the Table view (second icon on the left), then click on the upper
            right on "columns", remove all selected columns by clicking on "Hide all", then click
            on the left list on "ID", "Job Type", "max_depth", "n_estimators", "mae" and "r2".
            Click on "Close". Now in the table view you can click on the "mae" column
            on the three little dots, then select "Sort asc". This will sort the runs by ascending
            Mean Absolute Error (best result at the top).

When you have found the best job, click on its name. If you are interested you can explore some of the things we
tracked, for example the feature importance plot. You should see that the `name` feature has quite a bit of importance
(depending on your exact choice of parameters it might be the most important feature or close to that). The `name`
column contains the title of the post on the rental website. Our pipeline performs a very primitive NLP analysis 
based on [TF-IDF](https://monkeylearn.com/blog/what-is-tf-idf/) (term frequency-inverse document frequency) and can 
extract a good amount of information from the feature.

Go to the artifact section of the selected job, and select the 
`model_export` output artifact.  Add a ``prod`` tag to it to mark it as 
"production ready".

### Test
Use the provided step ``test_regression_model`` to test your production model against the
test set. Implement the call to this component in the `main.py` file. As usual, you can see the parameters in the
corresponding [MLproject](https://github.com/udacity/nd0821-c2-build-model-workflow-starter/blob/master/components/test_regression_model/MLproject) 
file. Use the artifact `random_forest_export:prod` for the parameter `mlflow_model` and the test artifact
`test_data.csv:latest` as `test_artifact`.

**NOTE**: This step is NOT run by default when you run the pipeline. In fact, it needs the manual step
of promoting a model to ``prod`` before it can complete successfully. Therefore, you have to
activate it explicitly on the command line:

```bash
> mlflow run . -P steps=test_regression_model
```

### Visualize the pipeline
You can now go to W&B, go the Artifacts section, select the model export artifact then click on the
``Graph view`` tab. You will see a representation of your pipeline.

### Release the pipeline
First, copy the best hyperparameters you found in your ``configuration.yml`` so they become the
default values. Then, go to your repository on GitHub and make a release. 
If you need a refresher, here are some [instructions](https://docs.github.com/en/github/administering-a-repository/managing-releases-in-a-repository#creating-a-release)
on how to release on GitHub.

Call the release ``1.0.0``:

![tag the release](images/tag-release-github.png "tag the release")

If you find problems in the release, fix them and then make a new release like ``1.0.1``, ``1.0.2``
and so on.

### Train the model on a new data sample

Let's now test that we can run the release using ``mlflow`` without any other pre-requisite. We will
train the model on a new sample of data that our company received (``sample2.csv``):

(be ready for a surprise, keep reading even if the command fails)
```bash
> mlflow run https://github.com/[your github username]/nd0821-c2-build-model-workflow-starter.git \
             -v [the version you want to use, like 1.0.0] \
             -P hydra_options="etl.sample='sample2.csv'"
```

**_NOTE_**: the file ``sample2.csv`` contains more data than ``sample1.csv`` so the training will
            be a little slower.

But, wait! It failed! The test ``test_proper_boundaries`` failed, apparently, there is one point
that is outside of the boundaries. This is an example of a "successful failure", i.e., a test that
did its job and caught an unexpected event in the pipeline (in this case, in the data).

We can fix this by adding these two lines in the ``basic_cleaning`` step just before saving the output 
to the csv file with `df.to_csv`:

```python
idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()
```
This will drop rows in the dataset that are not in the proper geolocation. 

Then commit your change, make a new release (for example ``1.0.1``) and retry (of course you need to use 
``-v 1.0.1`` when calling mlflow this time). Now the run should succeed and voit la', 
you have trained your new model on the new data.

## License

[License](LICENSE.txt)
