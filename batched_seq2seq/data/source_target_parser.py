#!/usr/bin/python
input_file = '../../lang8_english.txt'
f_out_src = open('lang8_english_src_1k.txt', 'w')
f_out_tgt = open('lang8_english_tgt_1k.txt', 'w')

count = 0
with open(input_file) as f:
    for line in f:
        if count == 1000:
            break
        items = line.split('\t')
        if items[0]!='' and items[1]!='' and items[0]!='\n' and items[1]!='\n':
            # write to output files
            f_out_src.write(items[0]+'\n')
            f_out_tgt.write(items[1])
            count += 1
f_out_src.close()
f_out_tgt.close()
