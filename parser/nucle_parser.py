import argparse
import os
import sys
import string
import re

#filepath = "/home/h2y/Desktop/annotations_alltypes.txt"

#Variables 
content = []
annotation = []
par = ""
err_type = ""
start_par = ""
start_off = ""
end_par = ""
end_off = ""
correction = ""

#T/F variables
skip = False
text = False
mistake = False
correct = False

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", dest="input_file", required=True, help="AESW file to be processed, in decoded format")
parser.add_argument("-o", "--output_file", dest="output_file", required=True, help="Where the output should be stored")
args = parser.parse_args()


with open(args.input_file, encoding = "ISO-8859-1") as f:

    for line in f:
        for word in line.split():

            #if end of doc, print 
            endofdoc = re.compile("</DOC>")
            if endofdoc.search(word):

                annotation.reverse()
                for cor in annotation: 
                    err_type = cor[0]
                    start_par = int(cor[1])
                    start_off = int(cor[2])
                    end_par = int(cor[3])
                    end_off = int(cor[4])
                    correction = cor[5]
                    start = " <del> "
                    end = " </del> <ins> "+correction+" </ins> "

                    #first match the par (start par and end par)
                    #then match the off (start off and end off)
                    #ignore if correction is empty

                    if correction and not correction.isspace():
                        content[end_par] = content[end_par][:end_off]+end+content[end_par][end_off:]
                        content[start_par] = content[start_par][:start_off]+start+content[start_par][start_off:]       

                # a paragraph with edited content
                # for each line
                #   if the line contains del or ins 
                #       for sentence1 -> content in del
                #       for sentence2 -> content in ins 
                #       [sentence1\tsentence2]
                #       print 

                for para in content:
                    for lines in para.split(". "):
                        q = re.compile("\?")
                        if q.search(lines):
                            lines = lines[:-1]
                            
                        first = ""
                        second = ""
                        dele = False
                        ins = False

                        for words in lines.split(" "):
                            #print (words)
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
                        #print(lines, file=open(args.output_file, "a"))
                        print(final, file=open(args.output_file, "a"))

                content = []
                annotation = []
            #if beginning of text, store content for revision later
            elif word == "<TEXT>":
                text = True
            elif word == "</TEXT>":
                text = False

            #if beginning of mistake, store the info to revise the content 
            elif word == "<MISTAKE":
                mistake = True
            elif word == "</MISTAKE>":
                cor_tuple = [err_type, start_par, start_off, end_par, end_off, correction]
                annotation.append(cor_tuple)
                err_type = ""
                start_par = ""
                start_off = ""
                end_par = ""
                end_off = ""
                correction = ""
                mistake = False                       

            #action
            elif text:
                tstart = re.compile("<TITLE>")
                tend = re.compile("</TITLE>")
                pstart = re.compile("<P>")
                pend = re.compile("</P>")
                if pend.search(word) or tend.search(word):
                    content.append(par)
                    #print (par)
                    par = ""
                elif not pstart.search(word) and not tstart.search(word):
                    par += word + " "

            elif mistake:
                startpar = re.compile("start_par=")
                startoff = re.compile("start_off=")
                endpar = re.compile("end_par=")
                endoff = re.compile("end_off=")
                errtype = re.compile("<TYPE>")
                startcorrection = re.compile("<CORRECTION>")
                endcorrection = re.compile("</CORRECTION>")

                if startpar.search(word):
                    word0 = word.replace("\"", "")
                    word0 = word0.replace("start_par=", "")
                    start_par = int(word0)
                    #print ("====start_par===")
                    #print (start_par)
                elif startoff.search(word):
                    word0 = word.replace("\"", "")
                    word0 = word0.replace("start_off=", "")
                    start_off = int(word0)
                    #print ("====start_off====")
                    #print (start_off)
                elif endpar.search(word):
                    word0 = word.replace("\"", "")
                    word0 = word0.replace("end_par=", "")
                    end_par = int(word0)
                    #print ("====end_par====")
                    #print (end_par)
                elif endoff.search(word):
                    word0 = word.replace("\"", "")
                    word0 = word0.replace("end_off=", "")
                    word0 = word0.replace(">", "")
                    end_off = int(word0)
                    #print ("====end_off====")
                    #print (end_off)
                elif errtype.search(word):
                    word0 = word.replace("<TYPE>", "")
                    word0 = word0.replace("</TYPE>", "")
                    err_type = word0
                    #print ("====err_type====")
                    #print (err_type)

                #enter correction
                elif startcorrection.search(word):
                    word0 = word.replace("<CORRECTION>", "")

                    #if only one word
                    if endcorrection.search(word):
                        word0 = word0.replace("</CORRECTION>", "")
                        correction += word0
                        #print ("====correction====")
                        #print (correction)
                    #if more than one word
                    else:
                        correct = True
                        correction += word0
                        correction += " "

                elif correct:
                    if endcorrection.search(word):
                        word0 = word.replace("</CORRECTION>", "")
                        correction += word0
                        correct = False
                        #print ("====correction====")
                        #print (correction)
                    else:
                        correction += word
                        correction += " "