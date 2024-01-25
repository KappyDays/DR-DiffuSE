import argparse

argsparser = argparse.ArgumentParser()

argsparser.add_argument('--gpus', type=int, nargs='+')
argsparser.add_argument('--wpn', type=str)
args = argsparser.parse_args()
# print(args.gpus, type(args.gpus))
# print(args.wpn, type(args.wpn))
if args.wpn:
    print(args.wpn)
    print("하이")