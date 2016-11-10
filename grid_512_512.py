import output
# import math

import logging
import argparse
import time
import sys
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

RHO = 0.5
ETA = 2 * (10**-4)

DELTA = [(0,-1), (0, 1), (-1,0), (1,0)]
G = 0.75
N = 512

def is_valid(x, y):
    return x >= 0 and x < N and y >= 0 and y < N

CORNERS = [0, N-1, N**2 - N, N**2 - 1]

LOAD_THRESHOLD = 400
def send_array(array, *args, **kwargs):
    assert N < LOAD_THRESHOLD * 2

    if N > LOAD_THRESHOLD:
        comm.send(array[:LOAD_THRESHOLD], *args, **kwargs)
        comm.send(array[LOAD_THRESHOLD:], *args, **kwargs)
    else:
        comm.send(array, *args, **kwargs)


def recv_array(*args, **kwargs):
    if N > LOAD_THRESHOLD:
        data1 = comm.recv(*args, **kwargs)
        data2 = comm.recv(*args, **kwargs)

        return np.hstack((data1, data2))
    else:
        return comm.recv(*args, **kwargs)


########################################################################################################

def test_comm():
    if rank == 0:
        data = np.array([i for i in xrange(512)])
        send_array(data, dest = 1)
        send_array(data, dest = 1)

        data = recv_array(source = 1)
        print "Received {} with {}".format(data.shape, data[:10])
    else:
        data = np.array([2 for _ in xrange(512)])
        send_array(data, dest = 0)

        data = recv_array(source = 0)
        print "1 Received {} with {}".format(data.shape, data[:10])

        data = recv_array(source = 0)
        print "1 Received {} with {}".format(data.shape, data[:10])

########################################################################################################
########################################################################################################

class Block(object):
    def __init__(self, block_size):
        super(Block, self).__init__()

        self.block_size = block_size
        self.val2 = np.zeros((block_size + 2, N))
        self.val1 = np.zeros((block_size + 2, N))
        self.new_val = np.zeros((block_size + 2, N))
        
    def update_inner(self, start_row, end_row):
        assert start_row <= end_row

        # Update the inner values
        for row in xrange(start_row, end_row):
            for col in xrange(1, N - 1):
                neighbors = [self.val1[row + d[0]][col + d[1]] for d in DELTA]

                self.new_val[row][col] = (sum(neighbors) -4 * self.val1[row][col]) * RHO + 2 * self.val1[row][col] - (1 - ETA) * self.val2[row][col]
                self.new_val[row][col] = self.new_val[row][col] / (1 + ETA)

    def update_left_right_border(self, start_row, end_row):
        assert start_row <= end_row

        # Update border values
        for row in xrange(start_row, end_row):
            self.new_val[row][0] = G * self.new_val[row][1] # Left border
            self.new_val[row][-1]  = G * self.new_val[row][-2] # Right border

    def update(self):
        self.val2 = self.val1
        self.val1 = self.new_val

        self.new_val = np.zeros((self.block_size + 2, N))

########################################################################################################
########################################################################################################
########################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train and evaluate model')
    parser.add_argument('iteration_count', metavar='iteration_count', type = int, help='Number of iteration')
    parser.add_argument('-d', '--debug', dest = 'debug', default = False, action = 'store_true', help = 'Enable debug')
    parser.add_argument('-s', '--square', dest = 'square', default = False, action = 'store_true', help = 'Calculate and print out the root mean square error.')
    parser.add_argument('-t', '--time', dest = 'time', default = False, action = 'store_true', help = 'Print out runtime.')
    args = parser.parse_args()

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    iteration_count = args.iteration_count
    block_size = N / nprocs

    float_formatter = lambda x: "%.5f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    logging.basicConfig(format = '[%(asctime)s][%(levelname)s][%(lineno)d] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S', level = logging.INFO if not args.debug else logging.DEBUG)

    def get_top_row(rank):
        return rank * block_size

    def get_bot_row(rank):
        return get_top_row(rank + 1) - 1

    def is_top_block(rank):
        return rank == 0

    def is_bot_block(rank):
        return rank == nprocs - 1

    block = Block(block_size)

    if rank == nprocs / 2:
        if nprocs == 1:
            block.val1[N/2 + 1][N/2] = 1
        else:
            block.val1[1][N/2] = 1

    squares = 0
    start = time.time()

    for iteration in xrange(iteration_count):
        if nprocs == 1:
            block.update_inner(2, N - 1)
            block.update_left_right_border(2, N - 1)

            # Update top and bot border
            for col in xrange(1, N - 1):
                block.new_val[1][col] = G * block.new_val[2][col] # Top
                block.new_val[-2][col] = G * block.new_val[-3][col] # Bot

            # Update corner
            # Left corner
            block.new_val[1][0] = G * block.new_val[2][0] # Top left
            block.new_val[-2][0] = G * block.new_val[-3][0] # Bot left

            # Right corner
            block.new_val[1][N-1] = G * block.new_val[1][N-2] # Top right
            block.new_val[-2][N-1] = G * block.new_val[-2][N-2] # Bot right
        else:
            if is_top_block(rank):
                logging.debug("From %s to %s with dimension %s" % (rank, rank + 1, block.val1[-2].shape))
                send_array(block.val1[-2], dest = rank + 1)
                logging.debug("Finished sending from %s to %s" % (rank, rank + 1))
                block.val1[-1] = recv_array(source = rank + 1)
                logging.debug("Me is {} received from {}".format(rank, rank + 1))

                # Update inner values
                block.update_inner(2, block_size + 1)

                # Update border values
                block.update_left_right_border(2, block_size + 1)

                # Update top border
                for col in xrange(1, N - 1):
                    block.new_val[1][col] = G * block.new_val[2][col]

                # Update corners
                # Left corner
                block.new_val[1][0] = G * block.new_val[2][0]

                # Right corner
                block.new_val[1][N-1] = G * block.new_val[1][N-2]

            elif is_bot_block(rank):
                logging.debug("From %s to %s with dimension %s" % (rank, rank - 1, block.val1[1].shape))
                send_array(block.val1[1], dest = rank - 1)
                logging.debug("Finished sending from %s to %s" % (rank, rank - 1))
                block.val1[0] = recv_array(source = rank - 1)
                logging.debug("Me is {} received from {}".format(rank, rank - 1))

                # Update inner values
                block.update_inner(1, block_size)

                # Update border values
                block.update_left_right_border(1, block_size)

                # Update bot border
                for col in xrange(1, N - 1):
                    block.new_val[-2][col] = G * block.new_val[-3][col]

                # Update corners
                # Left corner
                block.new_val[-2][0] = G * block.new_val[-3][0]

                # Right corner
                block.new_val[-2][N-1] = G * block.new_val[-2][N-2]
            else:
                logging.debug("Start sending from {} to {} and {}....".format(rank, rank - 1, rank + 1))
                send_array(block.val1[1], dest = rank - 1)
                logging.debug("Finished sending from {} to {}....".format(rank, rank - 1))
                send_array(block.val1[-2], dest = rank + 1)
                logging.debug("Finished sending from {} to {}....".format(rank, rank + 1))

                logging.debug("Waiting from {} and {}".format(rank - 1, rank + 1))
                block.val1[0] = recv_array(source = rank - 1)
                logging.debug("I am {} and received from {}".format(rank, rank - 1))
                block.val1[-1] = recv_array(source = rank + 1)
                logging.debug("I am {} and received from {}".format(rank, rank + 1))
                logging.debug("Finished receiving...")

                # Update the inner values
                block.update_inner(1, block_size + 1)

                # Update border values
                block.update_left_right_border(1, block_size + 1)

        block.update()

        if rank == nprocs / 2:
            to_print = block.val1[1][N/2] if nprocs != 1 else block.val1[N/2 + 1][N/2]

            if args.debug:
                logging.info("-----> {}".format(to_print))
            else:
                print "{},".format(to_print)

            if args.square:
                squares += (to_print - output.output[iteration]) ** 2

    if args.square and rank == nprocs / 2:
        print "Root mean squares: {} and is it good? {}".format(squares / iteration_count, squares / iteration_count < 0.00001)

    comm.Barrier()
    if rank == 0 and args.time:
        logging.info("With {} core(s) and iteration count = {}, took {}s".format(nprocs, iteration_count, time.time() - start))