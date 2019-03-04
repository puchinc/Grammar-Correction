import argparse
import os
import sys
import string
import re

#filepath = "./aesw2016(v1.2)_test.xml"
#output_file = "./aesw_test.txt"

#Variables 
content = []
allcontent = []

#T/F variables
dele = False
ins = False
sentence = False
revise = False

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", dest="input_file", required=True, help="AESW file to be processed, in decoded format")
parser.add_argument("-o", "--output_file", dest="output_file", required=True, help="Where the output should be stored")
args = parser.parse_args()

with open(args.input_file, encoding = "ISO-8859-1") as f:

    for line in f:
        for word in line.split():
            endofdoc = re.compile("</DOC>")
            startsentence = re.compile("sid=")
            endofsentence = re.compile("</sentence>")
            startdelete = re.compile("<del>")
            enddelete = re.compile("</del>")
            startins = re.compile("<ins>")
            endins = re.compile("</ins>")

            #if beginning of sentence, store the info to revise the content 
            if startsentence.search(word):
                sentence = True
                # add the word at the beginning of the sentence
                ws = re.split("\">", word)
                for a in ws:
                    if not startsentence.search(a):
                        a = re.sub("<", " <", a)
                        a = re.sub(">", "> ", a)
                        sep = re.split(" ", a)
                        for b in sep:
                            content.append(b)
            elif endofsentence.search(word):
                sentence = False       
                word = re.sub("<", " <", word)
                word = re.sub(">", "> ", word)
                sep = re.split(" ", word)
                for a in sep:
                    if a != "</sentence>":
                        content.append(a)
            elif sentence:
                #print (word)
                if startdelete.search(word) or startins.search(word) or enddelete.search(word) or endins.search(word): 
                    revise = True
                    word = re.sub("<", " <", word)
                    word = re.sub(">", "> ", word)
                    sep = re.split(" ", word)
                    for a in sep:
                        content.append(a)
                    '''
                    revise = True
                    add = ""
                    tag = ""
                    for a in word:
                        if a == "<":
                            if add:
                                content.append(add)
                                add = ""
                            tag+=a
                        elif a == ">":
                            tag+=a
                            content.append(tag)
                            tag = ""
                        elif tag=="":
                            add+=a
                        else:
                            tag+=a
                    content.append(add)
                    '''
                else:
                    content.append(word)   

            #action
            # not in a sentence but sentence has word and has ins or del
            if not sentence and content:
                if revise:
                    first = ""
                    second = ""
                    dele = False
                    ins = False
                    
                    for words in content:
                        if words == "<del>":
                            dele = True
                        elif words == "<ins>":
                            ins = True
                        elif words == "</del>":
                            dele = False
                        elif words == "</ins>":
                            ins = False
                        elif not dele and not ins:
                            first+=words
                            first+=" "
                            second+=words
                            second+=" "
                        elif dele and not ins:
                            first+=words
                            first+=" "
                        elif ins and not dele:
                            second+=words
                            second+=" "
                    
                    final = ""
                    first = re.sub(' +', ' ', first)
                    second = re.sub(' +', ' ', second)
                    final+=first[:-1]
                    final+="\t"
                    final+=second[:-1]
                    allcontent.append(final)
                    #print(final)
                content = []
                revise = False

with open(args.output_file, "w+") as out:
    for line in allcontent:
        out.write(line+"\n")