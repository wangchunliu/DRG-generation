import sys
import re


def file_to_list(in_file):
    '''Write list to file'''
    lst = []
    with open(in_file, "r") as in_f:
        for line in in_f:
            lst.append(line.strip())
    return lst

### 把alignment转化为字典
def trans_dic(align_line):
    align_dic = {}
    align_list = align_line.split(' *** ')
    for i, term in enumerate(align_list):
        align_dic[term.split('|||')[0]] = term.split('|||')[1]
    return align_dic

if __name__ == "__main__":
    ## input1 is generator's output sentence file, input2 is algiment file
    ## output is de_anonymized sentence data.
    input_f1, input_f2, output_f = sys.argv[1], sys.argv[2], sys.argv[3]
    sens = file_to_list(input_f1)
    aligns = file_to_list(input_f2)
    with open(output_f,'w') as outfile:
        for i, sen in enumerate(sens):
            if len(aligns[i]):
                align_dic = trans_dic(aligns[i])
                for key, value in align_dic.items():
                    try:  # two methods to get anonymized sentences
                        sen = re.sub(key, value, sen)
                    except:
                        sen = sen.replace(key, value)
                outfile.write(sen + '\n')
            else:
                outfile.write(sen + '\n')

