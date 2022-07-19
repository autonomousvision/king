#!/usr/bin/env bash

# Download regular data for AIM-BEVs
mkdir aim_bev_regular_data
cd aim_bev_regular_data
wget https://s3.eu-central-1.amazonaws.com/avg-projects/king/king_dataset.zip
unzip king_dataset.zip
rm king_dataset.zip
cd ..