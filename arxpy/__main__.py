"""Top-level script environment."""
import argparse
from arxpy.ciphers.simeck32_64 import Simeck32_64
from arxpy.ciphers.simeck48_96 import Simeck48_96
from arxpy.ciphers.simeck64_128 import Simeck64_128
from arxpy.ciphers.simon32_64 import Simon32_64
from arxpy.ciphers.simon48_72 import Simon48_72
from arxpy.ciphers.simon48_96 import Simon48_96
from arxpy.ciphers.simon64_128 import Simon64_128
from arxpy.ciphers.simon64_96 import Simon64_96
from arxpy.ciphers.speck32_64 import Speck32_64
from arxpy.ciphers.speck48_72 import Speck48_72
from arxpy.ciphers.speck48_96 import Speck48_96
from arxpy.ciphers.speck64_128 import Speck64_128
from arxpy.ciphers.speck64_96 import Speck64_96
from arxpy.diffcrypt.difference import XorDiff, RXDiff
from arxpy.diffcrypt.itercipher import OptimalRelatedKeyCh


list_ciphers = {
    "Simeck32_64": Simeck32_64, "Simeck48_96": Simeck48_96,
    "Simeck64_128": Simeck64_128,
    "Simon32_64": Simon32_64, "Simon48_72": Simon48_72, "Simon48_96": Simon48_96,
    "Simon64_96": Simon64_96, "Simon64_128": Simon64_128,
    "Speck32_64": Speck32_64, "Speck48_72": Speck48_72, "Speck48_96": Speck48_96,
    "Speck64_96": Speck64_96, "Speck64_128": Speck64_128
}

parser = argparse.ArgumentParser()
parser.add_argument("cipher", choices=list(list_ciphers.keys()))
parser.add_argument("difference", choices=["XOR", "RX"])
parser.add_argument("-f", "--filename")
parser.add_argument("-s", "--start_rounds", type=int, default=1)
parser.add_argument("-e", "--end_rounds", type=int, default=31)

args = parser.parse_args()

cipher = list_ciphers[args.cipher]

if args.difference == "XOR":
    diff_type = XorDiff
elif args.difference == "RX":
    diff_type = RXDiff

if args.filename:
    filename = args.filename
else:
    filename = None

start = args.start_rounds
end = args.end_rounds

OptimalRelatedKeyCh(cipher, diff_type, filename, start, end)
