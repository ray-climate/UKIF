#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    submit_slurm_prepare_job.py
# @Author:      Dr. Rui Song
# @Email:       rui.song@physics.ox.ac.uk
# @Time:        04/11/2024 21:23

# generate_dates.py
import os

MODIS_dir = './modis_pm25_data_crop'

date_list = []
for file in os.listdir(MODIS_dir):
    if file.endswith('.nc'):
        YYYYMMDD = file.split('_')[3]
        date_list.append(YYYYMMDD)

date_list = sorted(set(date_list))

for date in date_list:
    print(date)
    os.sysmtem(f'sbatch slurm_prepare_date_i.sh {date}')
