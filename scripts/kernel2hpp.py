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


def main():
    parser = ArgumentParser(
        prog="kernel2hpp",
        description="Convert OpenCL .cl kernels to C++ .hpp headers to embed kernel code directly in your projects")

    parser.add_argument("-i", "--input", dest="input",
                        required=True, help="input kernel", metavar="")

    parser.add_argument("-o", "--output", dest="output",
                        help="output header", metavar="")

    args = parser.parse_args()

    name_without_ext = Path(args.input).with_suffix("").name
    with open(args.input) as input:
        kernel_text = input.read()

    opts = re.findall(r"@(\w+).*(\[.*\])", kernel_text)
    optmap = {}
    for i in opts:
        optmap[i[0]] = json.loads(i[1])

    if "kernel" not in optmap:
        optmap["kernel"] = str(name_without_ext)
    else:
        optmap["kernel"] = optmap["kernel"][0]

    if (args.output is None):
        args.output = "./"

    oput_path = Path(args.output)
    if (oput_path.is_dir()):
        output = str(
            Path(args.output + optmap["kernel"]).with_suffix(".hpp"))
    else:
        output = args.output

    header_text = "#include <CL/opencl.hpp>\n#include <string>\n#include <utils.hpp>\n\n"
    header_text += "struct {} {{ \n".format(optmap["kernel"])

    header_text += "\tusing functor_type = cl::KernelFunctor<{}>;\n\n".format(
        ", ".join(optmap["signature"]))

    source_args = []
    if "macros" not in optmap:
        optmap["macros"] = {}

    for i in optmap["macros"]:
        source_args.append("{} {}_param".format(i["type"], i["name"]))

    header_text += "\tstatic std::string source({}) {{\n\t\tstatic const std::string {}_source = R\"(\n{})\";\n\n".format(", ".join(source_args), optmap["kernel"],
                                                                                                                          kernel_text)
    for i in optmap["macros"]:
        header_text += "\t\tauto {}_macro_def = clutils::kernel_define(\"{}\", {}_param);\n".format(
            i["name"], i["name"], i["name"])

    header_text += "\t\treturn "

    names = []
    for i in optmap["macros"]:
        names.append("{}_macro_def".format(i["name"]))
    names.append("{}_source".format(optmap["kernel"]))

    header_text += " + ".join(names) + ";\n"

    header_text += "\t}\n\n"
    header_text += "\tstatic std::string entry() {{ return \"{}\"; }}\n".format(
        optmap["entry"][0])

    header_text += "};\n"

    with open(output, "w") as oput:
        oput.write(header_text)

    print(optmap)


main()
