import time
import sys
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

RHO = 0.5
ETA = 2 * (10**-4)

DELTA = [(0,-1), (0, 1), (-1,0), (1,0)]
G = 0.75
N = 4

def is_valid(x, y):
    return x >= 0 and x < N and y >= 0 and y < N

CORNERS = [0, N-1, N**2 - N, N**2 - 1]

if __name__ == "__main__":
    nprocs = comm.Get_size()
    rank = comm.Get_rank()

    iteration_count = int(sys.argv[1])
    block_size = N / nprocs

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

    if rank == 1:
    	val1[1][2] = 1

    top_row = get_top_row(rank)
    bot_row = get_bot_row(rank)

    def update_inner(start_row, end_row):
        # Update the inner values
        for row in xrange(start_row, end_row):
            for col in xrange(1, N - 1):
                neighbors = [val1[row + d[0]][col + d[1]] for d in DELTA]

                new_val[row][col] = (sum(neighbors) -4 * val1[row][col]) * RHO + 2 * val1[row][col] - (1 - ETA) * val2[row][col]
                new_val[row][col] /= (1 + ETA)

    def update_left_right_border(start_row, end_row):
        # Update border values
        for row in xrange(start_row, end_row):
            new_val[row][0] = G * new_val[row][1] # Left border
            new_val[row][-1]  = G * new_val[row][-2] # Right border

    for iteration in xrange(iteration_count):
        if is_top_block(rank):
            print "From %s to %s" % (rank, rank + 1)
            comm.send(val1[-2], dest = rank + 1, tag = 1)
            print "Finished sending from %s to %s" % (rank, rank + 1)
            val1[-1] = comm.recv(source = rank + 1, tag = 1)

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

            # print "Done top block"
        elif is_bot_block(rank):
            # print "From %s to %s" % (rank, rank - 1)
            comm.send(val1[1], dest = rank - 1, tag = 1)
            print "Finished sending from %s to %s" % (rank, rank - 1)
            val1[0] = comm.recv(source = rank - 1, tag = 1)

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
            # print "Done bot block"
        else:
            comm.send(val1[1], dest = rank - 1, tag = 1)
            comm.send(val1[-2], dest = rank + 1, tag = 1)

            print "Waiting from {} and {}".format(rank - 1, rank + 1)
            val1[0] = comm.recv(source = rank - 1, tag = 1)
            val1[-1] = comm.recv(source = rank + 1, tag = 1)
            print "Finished receiving..."

            # Update the inner values
            update_inner(1, block_size + 1)

            start = time.time()
            print "Started working..."
            # Update border values
            for row in xrange(1, block_size + 1):
                new_val[row][0] = G * new_val[row][1] # Left border
                new_val[row][-1]  = G * new_val[row][-2] # Right border

            print "Took %s" % (time.time() - start)

        val2 = val1
        val1 = new_val

        if rank == (nprocs / 2) + 1:
            top_row = get_top_row(rank) + 1

            print val1[N / 2 - top_row][N/2]