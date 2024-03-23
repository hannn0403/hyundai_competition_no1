from solver import Solver
from config import args


if __name__ == "__main__":
    solver = Solver(config=args)
    solver.build()

    if args.mode == "train":
        solver.train()
    elif args.mode == "test":
        solver.test()
