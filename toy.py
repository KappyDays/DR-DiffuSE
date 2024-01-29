import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpus', type=int, default=[], nargs="+", help='set gpus')
parser.add_argument('--wpn', type=str)
parser.add_argument('--from_base', action='store_true', default=True, help='for refiner')
args = parser.parse_args()
# print(args.gpus, type(args.gpus))
# print(args.wpn, type(args.wpn))
print(args)
# print(args.gpus)
# print("하이")