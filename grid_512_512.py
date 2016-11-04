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
    assert iteration_count < LOAD_THRESHOLD * 2

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Train and evaluate model')
    parser.add_argument('iteration_count', metavar='iteration_count', type = int, help='Number of iteration')
    parser.add_argument('-d', '--debug', dest = 'debug', default = False, action = 'store_true', help = 'Enable debug')
    args = parser.parse_args()

    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    iteration_count = args.iteration_count
    block_size = N / nprocs

    logging.basicConfig(format = '[%(asctime)s][%(levelname)s][%(filename)s][%(lineno)d] - %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S', level = logging.INFO if not args.debug else logging.DEBUG)

    # test_comm()

    def get_top_row(rank):
        return rank * block_size

    def get_bot_row(rank):
        return get_top_row(rank + 1) - 1

    def is_top_block(rank):
        return rank == 0

    def is_bot_block(rank):
        return get_bot_row(rank) == N - 1

    val2 = np.zeros((block_size + 2, N))
    val1 = np.zeros((block_size + 2, N))
    new_val = np.zeros((block_size + 2, N))

    if rank == 0:
    	val1[3][2] = 1

    def update_inner(start_row, end_row):
        assert start_row <= end_row

        # Update the inner values
        for row in xrange(start_row, end_row):
            for col in xrange(1, N - 1):
                neighbors = [val1[row + d[0]][col + d[1]] for d in DELTA]

                new_val[row][col] = (sum(neighbors) -4 * val1[row][col]) * RHO + 2 * val1[row][col] - (1 - ETA) * val2[row][col]
                new_val[row][col] /= (1 + ETA)

    def update_left_right_border(start_row, end_row):
        assert start_row <= end_row

        # Update border values
        for row in xrange(start_row, end_row):
            new_val[row][0] = G * new_val[row][1] # Left border
            new_val[row][-1]  = G * new_val[row][-2] # Right border


    for iteration in xrange(iteration_count):
        if is_top_block(rank):
            logging.debug("From %s to %s with dimension %s" % (rank, rank + 1, val1[-2].shape))
            send_array(val1[-2], dest = rank + 1)
            logging.debug("Finished sending from %s to %s" % (rank, rank + 1))
            val1[-1] = recv_array(source = rank + 1)
            logging.debug("Me is {} received from {}".format(rank, rank + 1))

            # Update inner values
            update_inner(2, block_size + 1)

            # Update border values
            update_left_right_border(2, block_size + 1)

            # Update top border
            for col in xrange(1, N - 1):
                new_val[1][col] = G * new_val[2][col]

            # Update corners
            # Left corner
            new_val[1][0] = G * new_val[2][0]

            # Right corner
            new_val[1][N-1] = G * new_val[1][N-2]

        elif is_bot_block(rank):
            logging.debug("From %s to %s with dimension %s" % (rank, rank - 1, val1[1].shape))
            send_array(val1[1], dest = rank - 1)
            logging.debug("Finished sending from %s to %s" % (rank, rank - 1))
            val1[0] = recv_array(source = rank - 1)
            logging.debug("Me is {} received from {}".format(rank, rank - 1))

            # Update inner values
            update_inner(1, block_size)

            # Update border values
            update_left_right_border(1, block_size)

            # Update bot border
            for col in xrange(1, N - 1):
                new_val[-2][col] = G * new_val[-3][col]

            # Update corners
            # Left corner
            new_val[-2][0] = G * new_val[-3][0]

            # Right corner
            new_val[-2][N-1] = G * new_val[-2][N-2]
        else:
            logging.debug("Start sending from {} to {} and {}....".format(rank, rank - 1, rank + 1))
            send_array(val1[1], dest = rank - 1)
            logging.debug("Finished sending from {} to {}....".format(rank, rank - 1))
            send_array(val1[-2], dest = rank + 1)
            logging.debug("Finished sending from {} to {}....".format(rank, rank + 1))

            logging.debug("Waiting from {} and {}".format(rank - 1, rank + 1))
            val1[0] = recv_array(source = rank - 1)
            logging.debug("I am {} and received from {}".format(rank, rank - 1))
            val1[-1] = recv_array(source = rank + 1)
            logging.debug("I am {} and received from {}".format(rank, rank + 1))
            logging.debug("Finished receiving...")


            # Update the inner values
            update_inner(1, block_size + 1)

            # Update border values
            update_left_right_border(1, block_size + 1)

            start = time.time()
            logging.debug("Started working...")
            # Update border values
            for row in xrange(1, block_size + 1):
                new_val[row][0] = G * new_val[row][1] # Left border
                new_val[row][-1]  = G * new_val[row][-2] # Right border

            logging.debug("Took %s" % (time.time() - start))

        comm.Barrier()
        val2 = val1
        val1 = new_val

        if rank == nprocs / 2:
            logging.info("----------------------------------------> {} - {}".format(val1[1][N/2], val2[1][N/2]))