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

Results for each experiment will be written out to a csv file in a directory of the form `results-{uniform | static | dynamic}-{dataset}`. A summary of the results will also be printed to the console. The output from running the experiments using 100 trials should look similar to the following:
```
(venv) ubuntu@ip-172-31-27-186:~/InQuest$ ./run_baselines_and_inquest.sh -t 100
------ Running baselines + InQuest on dataset customer-support ------
--- customer-support uniform ---
100%|███████████████████████████████████████████████████████| 1000/1000 [10:42<00:00,  1.56it/s]
  oracle limit  500 mean rmse error: 0.02618
  oracle limit 1000 mean rmse error: 0.01750
  oracle limit 1500 mean rmse error: 0.01426
  oracle limit 2000 mean rmse error: 0.01227
  oracle limit 2500 mean rmse error: 0.01117
  oracle limit 3000 mean rmse error: 0.00991
  oracle limit 3500 mean rmse error: 0.00924
  oracle limit 4000 mean rmse error: 0.00893
  oracle limit 4500 mean rmse error: 0.00760
  oracle limit 5000 mean rmse error: 0.00856
--- customer-support static ---
100%|███████████████████████████████████████████████████████| 1000/1000 [10:56<00:00,  1.52it/s]
  oracle limit  500 mean rmse error: 0.02209
  oracle limit 1000 mean rmse error: 0.01476
  oracle limit 1500 mean rmse error: 0.01302
  oracle limit 2000 mean rmse error: 0.01011
  oracle limit 2500 mean rmse error: 0.00905
  oracle limit 3000 mean rmse error: 0.00851
  oracle limit 3500 mean rmse error: 0.00800
  oracle limit 4000 mean rmse error: 0.00732
  oracle limit 4500 mean rmse error: 0.00747
  oracle limit 5000 mean rmse error: 0.00672
--- customer-support InQuest ---
100%|███████████████████████████████████████████████████████| 1000/1000 [11:36<00:00,  1.43it/s]
  oracle limit  500 mean rmse error: 0.01980
  oracle limit 1000 mean rmse error: 0.01357
  oracle limit 1500 mean rmse error: 0.00920
  oracle limit 2000 mean rmse error: 0.00835
  oracle limit 2500 mean rmse error: 0.00668
  oracle limit 3000 mean rmse error: 0.00704
  oracle limit 3500 mean rmse error: 0.00686
  oracle limit 4000 mean rmse error: 0.00551
  oracle limit 4500 mean rmse error: 0.00493
  oracle limit 5000 mean rmse error: 0.00509
------ Running baselines + InQuest on dataset archie11 ------
--- archie11 uniform ---
...
...continues for other datasets
...
```


