import pandas as pd
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-o", "--output_path", type=str)

args = vars(parser.parse_args())
outfile = args['output_path']

data = sys.stdin.readlines()  
combined_csv = pd.concat(
    [pd.read_csv(f.strip(), encoding='utf-8') for f in data], ignore_index=True)
combined_csv.sort_values(by=['iteration'], ascending=True, inplace=True)
combined_csv.to_csv(outfile, index=False)