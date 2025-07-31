#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import yaml


def load_config(file_path="config.yaml"):
    with open(file_path) as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def read_dataframe(file_path, target, quick_train):
    df = pd.read_csv(file_path)
    
    # Get categ and numeric columns
    categ_cols = [c for c in df.columns if df[c].dtype == 'object']
    num_cols = [c for c in df.columns if c not in categ_cols]
    num_cols.remove(target)

    for column in categ_cols:
        df[column] = df[column].astype('category')
    categ_cols.remove('policy_id')

    # Convert 'is_' columns to integer values
    is_cols = [c for c in df.columns if c.startswith('is_')]
    for column in is_cols:
        df[column] = df[column].map(dict(Yes=1, No=0))
        df[column] = df[column].astype('int16')
        num_cols.append(column)
        categ_cols.remove(column)

    if quick_train:
        for column in is_cols:
            num_cols.remove(column)
        df = df.sample(n=10000, random_state=42)

    df = df[num_cols + categ_cols + [target]]

    return df