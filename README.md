# InQuest
This repository contains the source code for InQuest. In this document we detail how to set up and reproduce our main experimental results in the paper.

## Setup
For this walkthrough we will use a `c5.24xlarge` EC2 instance running an Ubuntu AMI. Smaller instances may also suffice, although you will likely want to modify certain parameters that control the number of cores used and the number of trials per oracle budget. We discuss these settings in more detail in the following section.

Once you've acquired an instance from EC2 (or if you are running on a local UNIX machine) you may run the following steps to set up your environment:
```
# install python3.8
sudo apt update
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.8 python3.8-venv libpython3.8-dev

# create and activate a virtualenv
python3.8 -m venv venv
source venv/bin/activate

# clone this repository;
# note if you are doing this from a new EC2 instance you will
# likely need to set up new Github credentials
git clone git@github.com:stanford-futuredata/InQuest.git
cd InQuest

# install required Python packages
pip install -r requirements.txt
```

## Run Experiments
Once your environment is set up, you can run InQuest and the baseline algorithms on our six evaluation datasets by executing:
```
./run_baselines_and_inquest.sh [-t trials-per-oracle-limit | -n num-processes]
```
Running the script without any arguments will use a default of 1000 trials per oracle budget (i.e. oracle limit) parallelized over 48 cores. If you are running this on a smaller machine, I suggest you set `-n` to be the number of physical cores you have available. Similarly, running these experiments with fewer trials-per-oracle-limit (e.g. `-t 100`) can yield results much quicker.

Results for each experiment will be written out to a csv file in a directory of the form `results-{uniform | static | dynamic}-{dataset}`. A summary of the results will also be printed to the console. The output from running the experiments using 100 trials should look something like the following:
```

```


