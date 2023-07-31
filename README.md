# gpt-2-training-code
gpt-2 training on code120_train.csv

*This example will train a gpt-2 model ona code dataset from hugginface.

*To start this process a local python environment need to be created using the following command

conda env create -f environment.yml

*The attached file environment.yml contains all the good shit need for this to work, no questions asked warranty. 

*The next step is to create the *.csv file that will be used for training, this is done using the following command

python gpt-2_v12b_create_csv.py

*The *.csv file will be approx 64 Mb

*Last step is to start the training using 

python gpt-2_v12b.py

%------------------------------------
With current settings in gpt-2_v12b.py and a RTX6000, expected training time 13 hours.
Make sure to have 4GB Disk space available for the checkpoints that is stored every 500 itterations.
%------------------------------------
