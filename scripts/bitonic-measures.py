#!/usr/bin/python

# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <tsimmerman.ss@phystech.edu>, <alex.rom23@mail.ru> wrote this file.  As long as you
# retain this notice you can do whatever you want with this stuff. If we meet
# some day, and you think this stuff is worth it, you can buy us a beer in
# return.
# ----------------------------------------------------------------------------

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import re
import json
import subprocess
import matplotlib.pyplot as plt


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
    parser.add_argument("--lsz", dest="lsz",
                        help="Local memory size to use", metavar="")

    return parser.parse_args()



def execute_test (binname: str, kernel: str, n: int, lsz: int) -> str:
    args = (binname, "--kernel", kernel, "-n", str(n), "--lsz", str(lsz))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read().decode("utf-8")
    return output


def run_test_json_text (binname: str, kernel: str, n: int, lsz: int) -> str:
    output_text = execute_test(binname, kernel, n, lsz)
    output_numbers = list(map(int, re.findall(r'\d+', output_text)))
    json_source = f'''{{
\"test\" : {{
        \"lsz\" : {lsz},
        \"len\" : {output_numbers[0]},
        \"std_time\" : {output_numbers[1]},
        \"gpu_wall\" : {output_numbers[2]},
        \"gpu_pure\" : {output_numbers[3]}
}}
}}'''
    return json_source

def run_all_tests_for_kernel (binname: str, kernel: str, max_n: int, lsz: int) -> str:
    json_list = []
    for n in range(17, max_n + 1):
            json_list.append(run_test_json_text(binname, kernel, n, lsz))
    return f'\"{kernel}s\" : [' + ', \n'.join(json_list) + ']\n'

    

def run_all_tests(binname: str, kernels_list: list, max_n: int, max_lsz: int) -> str:
    json_source_list = []
    for kernel in kernels_list:
        json_source_list.append(run_all_tests_for_kernel(binname, kernel, max_n, max_lsz))
    return '{\n'+  ',\n'.join(json_source_list) + '}'


def write_to_measures_json_file(filename: str, json_obj: str)->None:
    with open(filename, "w") as output:
        output.write(json_obj)
        

def plot_measurements_of_kernel (filename: str, kernel_list: list, data)-> None:
    fig, ax = plt.subplots(figsize=(100, 50))
    ax.set_xscale('symlog', base=2)
    ax.grid()
    first = True
    lens, gpu_times, cpu_times = [], [], []

    for kernel in kernel_list:
        gpu_times.clear()
        lens.clear()
        lsz = data[kernel + 's'][0]["test"]["lsz"]
        for test in data[kernel + 's']:
            lens.append(test["test"]["len"])
            gpu_times.append(test["test"]["gpu_wall"] / 1000)
            if (first):
                cpu_times.append(test["test"]["std_time"] / 1000)
        plt.plot(lens, gpu_times, marker='o', label=f"{kernel} bitonic sort")
        first = False
    
    plt.plot(lens, cpu_times, marker='o', label="std::sort")
    
    plt.xlabel("Len of the test")
    plt.ylabel("Time spend, s")
    plt.title(f"local size = {lsz}")
    plt.legend()
    plt.show()
        
def plot_measurements (filename : str, kernel_list: list) -> None:
    with open(filename) as json_file:
        data = json.load(json_file)
    plot_measurements_of_kernel(filename, kernel_list, data)


def main():
    args = parse_cmd_args()
    kernel_list = ["local", "naive"]
    json_source = run_all_tests(args.input, kernel_list, int(args.max_n), int(args.lsz))
    write_to_measures_json_file(args.output, json_source)
    plot_measurements(args.output, kernel_list)



if (__name__ == "__main__"):
    main()

    
