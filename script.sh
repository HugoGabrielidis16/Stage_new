'python3 centralized_run.py --Dataset=MNIST --strategy=FedAvg --nbr_clients=7 --nbr_rounds=50 --accumulated_data=False --centralized_percentage=1 --directory_name=$1
'
python3 federated_run.py --Dataset=CIFAR10 --strategy=FedAvg --nbr_clients=7 --nbr_rounds=50 --accumulated_data=False --centralized_percentage=0.2 --directory_name=$1
