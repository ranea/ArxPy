# ArxPy 0.1

Tool to find optimal related-key chararacteristics.


```
$ python3 arxpy -h
usage: arxpy [-h] [-f FILENAME] [-s START_ROUNDS] [-e END_ROUNDS]
             {Simeck32_64,Simeck48_96,Simeck64_128,Simon32_64,Simon48_72,Simon48_96,Simon64_96,Simon64_128,Speck32_64,Speck48_72,Speck48_96,Speck64_96,Speck64_128}
             {XOR,RX}

positional arguments:
  {Simeck32_64,Simeck48_96,Simeck64_128,Simon32_64,Simon48_72,Simon48_96,Simon64_96,Simon64_128,Speck32_64,Speck48_72,Speck48_96,Speck64_96,Speck64_128}
  {XOR,RX}

optional arguments:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
  -s START_ROUNDS, --start_rounds START_ROUNDS
  -e END_ROUNDS, --end_rounds END_ROUNDS
```

## Dependencies

- python3
- sympy
- bidict
- pySMT with boolector

