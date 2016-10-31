import sys
from mpi4py import MPI

RHO = 0.5
ETA = 2 * (10**-4)

DELTA = [(0,-1), (0, 1), (-1,0), (1,0)]
G = 0.75
N = 4

# 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15

comm = MPI.COMM_WORLD

def is_valid(x, y):
    return x >= 0 and x < N and y >= 0 and y < N

def is_corner(rank):
    return rank in [0, N-1, N**2 - N, N**2 - 1]

def is_top_row(rank):
    return rank < N

def is_bot_row(rank):
    return rank > N**2 - N

def is_left_col(rank):
    return rank % N == 0

def is_right_col(rank):
    return (rank - N + 1) % N == 0

def is_border(rank):
    return is_top_row(rank) or is_bot_row(rank) or is_left_col(rank) or is_right_col(rank)

get_rank = lambda x, y : x * N + y
get_coordinate = lambda rank : (rank / N, rank % N)

def send_out(rank, my_val, tag):

    my_coordinate = get_coordinate(rank)
    neighbors = [(my_coordinate[0] + d[0], my_coordinate[1] + d[1]) for d in DELTA]

    for neighbor in neighbors:
        if not is_valid(*neighbor):
            continue

        comm.send(my_val, dest = get_rank(*neighbor), tag = tag)

if __name__ == "__main__":
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    my_coordinate = get_coordinate(rank)

    iteration_count = sys.argv[1]

    for iteration in xrange(1):
        val2 = 0
        val1 = 0 if my_coordinate != (2,2) else 1
        val = 0
        new_val = 0
        # print("The rank is %s %s, %s" % (str(rank), is_corner(rank), is_border(rank)))
        # Begin iteration
        send_out(rank, val1, tag = 1)


        if is_corner(rank):
            src = [(0,0), (N-1,0), (0,N-1), (N-1,N-1)]
            dest = [(1,0), (N-2, 0), (0, N-2), (N-1, N-2)]
            corner_dictionary = {x[0]: x[1] for x in zip(src, dest)}

            dest_rank = get_rank(*corner_dictionary[my_coordinate])
            new_val = G * comm.recv(source=dest_rank, tag = 0)
        elif is_border(rank):
            if is_top_row(rank):
                dest_rank = get_rank(1, my_coordinate[1])
            elif is_bot_row(rank):
                dest_rank = get_rank(N-2, my_coordinate[1])
            elif is_left_col(rank):
                dest_rank = get_rank(my_coordinate[0], 1)
            elif is_right_col(rank):
                dest_rank = get_rank(my_coordinate[0], N-2)

            new_val = G * comm.recv(source=dest_rank, tag = 0)
            send_out(rank, new_val, tag = 0)
        else:
            new_val = sum(comm.recv(source = get_rank(my_coordinate[0] + d[0], my_coordinate[1] + d[1]), tag = 1) for d in DELTA)
            new_val += -4 * val1
            new_val *= RHO
            new_val += 2 * val1 - (1 - ETA) * val2

            new_val /= (1 + ETA)
            send_out(rank, new_val, tag = 0)


        val2 = val1
        val1 = val
        val = new_val

        print("({}, {}): {}".format(my_coordinate[0], my_coordinate[1], val))
