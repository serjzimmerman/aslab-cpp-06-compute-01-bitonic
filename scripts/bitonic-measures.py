#!/usr/bin/python

# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
# retain this notice you can do whatever you want with this stuff. If we meet
# some day, and you think this stuff is worth it, you can buy us a beer in
# return.
# ----------------------------------------------------------------------------

from argparse import ArgumentParser
from pathlib import Path
import re
import json
import subprocess



def parse_cmd_args():
    parser = ArgumentParser(
        prog="bitonic-measures",
        description="Measure performance of our bitonic sort alorithms")

    parser.add_argument("-i", "--input", dest="input",
                        required=True, help="input binary file", metavar="")

    parser.add_argument("-o", "--output", dest="output",
                        help="raw data output file", metavar="")
    parser.add_argument("-n", dest="max_n",
                        help="Maximum power of two in len of test", metavar="")
    parser.add_argument("--lsz", dest="max_lsz",
                        help="Maximum local memory size to use", metavar="")

    return parser.parse_args()



def execute_test (binname: str, kernel: str, n: int, lsz: int) -> str:
    args = (binname, "--kernel", kernel, "-n", str(n), "--lsz", str(lsz))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    return output


def run_test_json_text (test_n: int, binname: str, kernel: str, n: int, lsz: int) -> str:
    output_text = execute_test(binname, kernel, n, lsz)
    output_numbers = list(map(int, re.findall(r'\d+', output_text)))
    json_source = f'''\"test{test_n}\" : {{
        \"kernel\" : \"{kernel}\",
        \"lsz\" : {lsz},
        \"len\" : {output_numbers[0]},
        \"std_time\" : {output_numbers[1]},
        \"gpu_wall\" : {output_numbers[2]},
        \"gpu_pure\" : {output_numbers[3]}
}}'''
    return json_source

def run_all_tests_json_list(binname: str, kernels_list: list, max_n: int, max_lsz: int) -> list:
    json_source_list = []
    test_n = 0
    for kernel in kernels_list:
        for n in range(15, max_n + 1):
            for lsz in [2**i for i in range(5, max_lsz+1)]:
                json_source_list.append(run_test_json_text(test_n,binname, kernel, n, lsz))
                test_n += 1
    return json_source_list

def run_all_tests_json_text(binname: str, kernels_list: list, max_n: int, max_lsz: int) -> str:
    json_list = run_all_tests_json_list(binname, kernels_list, max_n, max_lsz)
    json_source = '{\n' + ', \n'.join(json_list) + '\n}'
    return json_source


def write_to_measures_json_file(filename: str, json_obj: str)->None:
    with open(filename, "w") as output:
        output.write(json_obj)
        


def main():
    args = parse_cmd_args()
    json_source = run_all_tests_json_text(args.input, ["local"], int(args.max_n), int(args.max_lsz))
    write_to_measures_json_file(args.output, json_source)


if (__name__ == "__main__"):
    main()

    
