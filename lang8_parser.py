#!/usr/bin/python

# This script parses lang-8 data file (lang-8-20111007-L1-v2.dat) and write to a new file with the format: learner_sentence \t corrected_sentence \n
# reference: https://github.com/nusnlp/mlconvgec2018/blob/master/data/scripts/lang-8_scripts/extract.py
# example usage: python lang8_parser.py -i lang-8-20111007-L1-v2.dat -o lang8_parsed -l2 English 
# parsed file will be on lab machine 

import argparse
import os
import sys
import string 

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", dest="input_file", required=True, help="Lang8 file to be processed, in decoded format")
parser.add_argument("-o", "--output_dir", dest="output_dir", required=True, help="Path to output directory")
parser.add_argument("-l1","--native_language", dest="l1", help="Extract only for this native language")
parser.add_argument("-l2","--learning_language", dest="l2", required=True, help="Extract only for this learning language")
parser.add_argument("-split", dest="split_out", action="store_true", help="Enable this flag if output is to be split to different native languages")
args = parser.parse_args()

count = 0
count_output = 0
output_file = 'lang8_english'

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.l1 is None:
    l1dict = dict()
    f_combined_out=open(args.output_dir+"/" + output_file + ".txt",'w')
else:
    if args.l2 is None:
        print >> sys.stderr, "L2 should also be specfied"
        sys.exit()
    f_out = open(args.output_dir + "/" + os.path.basename(args.input_file) + "." + "processed.l1_" + args.l1,'w')

with open(args.input_file) as f:
    for line in f:
        l2= line.split(',')[2][1:-1] 
        l1= line.split(',')[3][1:-1]
        # If Learning Language is Specified and Learning Language = Required Learning Language
        if args.l2 is not None and args.l2 == l2:
            if args.l1 is None:
                if args.split_out:
                    if l1dict.has_key(l1): # To check if the native language file has been opened
                        f_out = l1dict[l1]
                        f_out.write(line)
                    else:
                        f_out = open(args.output_dir + "/" + os.path.basename(args.input_file) + "." + "processed.l1_" + l1,'w')
                        l1dict[l1] = f_out
                        f_out.write(line)
                else:
                    # delete tags 
                    line = line.replace('[f-red]','')
                    line = line.replace('[\\/f-red]','')
                    line = line.replace('[f-blue]','')
                    line = line.replace('[\\/f-blue]','')
                    line = line.replace('[sline]','')
                    line = line.replace('[\\/sline]','')
                    line = line.replace('[f-bold]','')
                    line = line.replace('[\\/f-bold]','')
                    # extract learner sentence and corrected sentence
                    learner_sen = line.split('[')[2][1:-2]
                    learner_sen_token = learner_sen.split('","')
                    correct_sen = line.split('[[')[1]
                    correct_line = correct_sen.split(',[')
                    index = 0 
                    # write learner sentence and corrected sentence to a new file with the format: learner_sentence \t corrected_sentence \n
                    for token in learner_sen_token:
                        if index >= len(correct_line):
                            break
                        if correct_line[index].strip(string.punctuation) != '' and correct_line[index].strip(string.punctuation) != ' ' and correct_line[index].strip(string.punctuation) != '\n':
                            correct_line[index] = correct_line[index].split('","')[0]
                            combined_sentence = token.strip(string.punctuation) + '\t' + correct_line[index].strip(string.punctuation)
                            combined_sentence = combined_sentence.replace(']]]','')
                            combined_sentence = combined_sentence.replace('\n','')
                            f_combined_out.write(combined_sentence + '\n') # a line for all index of correct_line.split
                            count_output += 1
                        index += 1
            # If Native Language is specified write that only.
            else:
                if args.l1 == l1:
                    f_out.write(line)
        # If no language is specified or only L1 is specified
        elif args.l2 is None:
            f_combined_out.write(line)
print(count_output)
if args.l1 is not None:
    f_out.close()

if args.l1 is None and args.split_out == True:
    for l1, f_out in l1dict.iteritems():
        f_out.close()