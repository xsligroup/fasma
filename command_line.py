from argparse import ArgumentParser, Namespace
from fasma.core import file_compressor as fc
from fasma.core import df_filters as dff
from fasma.core import printer
from pathlib import Path
import ast

parser = ArgumentParser()

parser.add_argument('filepath', help='Full directory path to file to be parsed', type=str)
parser.add_argument('-f', '--filename', help='Custom filename to save created files', type=str)
parser.add_argument('-s', '--save', help='Saves box object after parsing', action='store_true')
parser.add_argument('-p', '--print', help='Prints MO_Analysis after parsing', action='store_true')
parser.add_argument("--mo", help='MOs to be printed', type=str)
parser.add_argument("--atom", help='Dictionary of custom atom groupings', type=str)
args: Namespace = parser.parse_args()

box = fc.parse(args.filepath)
filename = Path(args.filepath).stem
if args.filename:
    filename = args.filename
if args.save:
    box.save(filename)
if args.print:
    mo = None
    atom = None
    if args.mo:
        mo = args.mo
    if args.atom:
        atom = ast.literal_eval(args.atom)
    data = box.generate_mo_analysis(full=True)
    data = dff.filter_mo_analysis(data, mo_list=mo, atoms=atom)
    printer.print_mo_analysis(filename, box, data)
