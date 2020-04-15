# Sweep

Utilites for hyperparameter sweeps, using [Weights and Biases](https://www.wandb.com/). These instructions are a minimial subset of [these docs](https://docs.wandb.com/sweeps).

To begin a sweep follow these steps:

1. Make sure wandb is installed:
   > pip install wandb
2. Login to wandb: 
    > wandb login
3. Initialize wandb (optional)
    > wandb init
4. When asked to choose a project make sure you  select `covasim`.  If you don't see a project with this name, instead select `Create New` and name your project **`covasim`**.
5. From the **root of this repo**, initialize a sweep.

    ```bash
    # Choose the yaml file that corresponds to the search strategy.
    wandb sweep sweep/sweep-random.yaml
    ```
    This command will print out a **sweep ID**. Copy that to use in the next step!


6. Launch agent(s)

    ```bash
    wandb agent your-sweep-id
    ```

    From [the docs](https://docs.wandb.com/sweeps/quickstart): 
    > You can run wandb agent on multiple machines or in multiple processes on the same machine, and each agent will poll the central W&B Sweep server for the next set of hyperparameters to run.
    
 ### Running sweeps in parallel
 
 You can poll the wandb parameter server with multiple agents to help tuning run more quickly.  The wandb hyperparameter server maintains a queue of parameters to test, which is consumed by agents when you run a `wandb agent <agent_id>`. 
 
For example, if you wanted to run two agents in parallel for sweep id `1234` (you get this from step 5 above) you would start two seperate processes (either a seperate terminal windows or background processes):
 
 ```bash
 wandb agent 1234
 ```
 In a seperate window repeat the exact same command
 
 ```bash
 wandb agent 1234
 ```
 
 Note: You can automate this with a bash script depending on your needs.
 
