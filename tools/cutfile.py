#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math

def make_dirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)

def get_total_lines(file_path):
	if not os.path.exists(file_path):
		return 0
	cmd = 'wc -l %s' % file_path
	return int(os.popen(cmd).read().split()[0])

def split_file_by_row(filepath, new_filepath, row_cnt, suffix_type='-d'):
    #tmp_dir = "/split_file_by_row/"
    #make_dirs(new_filepath)
    #make_dirs(new_filepath+tmp_dir)
    print row_cnt
    total_rows = get_total_lines(filepath)
    file_cnt = int(math.ceil(total_rows*1.0/row_cnt))
    command = "split -l %d %s -a 2 %s"%(row_cnt,filepath,suffix_type)#,r""+new_filepath+tmp_dir)
    print command
    os.system(command)
    return [new_filepath+fn for fn in new_filepath]


if __name__ == "__main__":
    try:
        import psyco
        psyco.profile()
    except ImportError:
        pass
    projectRoot = os.path.abspath('.')
    print projectRoot
    sourceFile = projectRoot+r"/../data/source.txt"
    print sourceFile
    outputsDir = projectRoot+r"/../data"
    split_file_by_row(sourceFile, outputsDir, get_total_lines(sourceFile)/10*8, "train.txt")
