#! /usr/bin/python

# prepare train, test, val data in csv file format
# Example Usage:
# python prepare_csv.py \
#       -i ../data/test/lang8_small.txt \
#       -train ../data/test/lang8_small_train.csv \
#       -train_r 0.6 \
#       -test ../data/test/lang8_small_test.csv \
#       -test_r 0.2 \
#       -val ../data/test/lang8_small_val.csv \
#       -val_r 0.2

import csv 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()        
    parser.add_argument('-i', '--input_file')                   
    parser.add_argument('-train', '--train_file')
    parser.add_argument('-test', '--test_file')
    parser.add_argument('-val', '--val_file')
    parser.add_argument('-train_r', '--train_ratio')
    parser.add_argument('-test_r', '--test_ratio') 
    parser.add_argument('-val_r', '--val_ratio')                
    args = parser.parse_args()                                      

    return args

def prepare_csv(input_file, file_names, data_ratio):
    num_lines = sum(1 for line in open(input_file))
    data_size = [int(i * num_lines) for i in data_ratio]
    data_type = 0
    count = 0
    csv = open(file_names[0], 'w')
    with open(input_file) as f:
        for line in f:
            if count == data_size[0]:
                csv = open(file_names[1], 'w')
            if count == data_size[0] + data_size[1]:
                csv = open(file_names[2], 'w')
            csv.write(line)
            count += 1

def main():
    args = parse_args()
    input_file = args.input_file
    train_file = args.train_file
    test_file = args.test_file
    val_file = args.val_file
    train_ratio = args.train_ratio
    test_ratio = args.test_ratio
    val_ratio = args.val_ratio

    file_names = [train_file, test_file, val_file]
    data_ratio = [float(train_ratio), float(test_ratio), float(val_ratio)]
    prepare_csv(input_file, file_names, data_ratio)

if __name__ == '__main__':
    main()
