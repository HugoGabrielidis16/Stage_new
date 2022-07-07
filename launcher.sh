#!/bin/bash

for i in range 5
do
	for dataset in IMDB 
	do
		for strategy in FedAvg FedYogi FedAdam FedAdagrad
		do
				python3 Launcher.py --Dataset=$dataset --strategy=$strategy --nbr_clients=8  --nbr_rounds=5
		done
	done
done


