# This is a distraction, but I'd like to write a headless version of minesweeper for fun.
# That's something I learned while doing leetcode, and trying to represent a chessboard in code for N Queens.
# Each tile can be in two states, revealed or unrevealed. revealed can have 10 values: mine neighbour count between 0 and 8, and mine.
# I think i don't need to worry toooooo much about performance because the bottleneck on this is the machine learning part, not the game part.
# To that end I think I will use numpy.

import numpy as np
import random

# init
n = 9 #nxn board
mine_count = 10
revealed = np.full((n,n), 9, dtype = np.uint8)




def create_underneath(mine_positions, n):
    base = np.zeros((n,n), dtype = np.int8)
    rows, cols = zip(*mine_positions)
    base[rows, cols] = 1

    padded = np.pad(base, 1, mode='constant', constant_values=0)
    
    nw = padded[:-2, :-2]
    n  = padded[:-2, 1:-1]
    ne = padded[:-2, 2:]
    w  = padded[1:-1, :-2]
    e  = padded[1:-1, 2:]
    sw = padded[2:, :-2]
    s  = padded[2:, 1:-1]
    se = padded[2:, 2:]

    neighbours = np.stack([nw, n, ne, w, e, sw, s, se], axis=-1)
    underneath = neighbours.sum(axis=-1)
    underneath[base==1] = -1
    return underneath



def reveal(revealed, r, c):
    if underneath[(r,c)] == -1:
        print("GAME_OVER")
        return
    if revealed[(r,c)] != 9:
        return

    revealed[(r,c)] = underneath[(r,c)]
    stack = [(r,c)]

    while stack:
        r, c = stack.pop()
        if underneath[(r,c)] == 0:
            neighbours = get_neighbours(r,c)
            for neighbour in neighbours:
                if revealed[neighbour] == 9:
                    revealed[neighbour] = underneath[neighbour]
                    if revealed[neighbour] == 0:
                        stack.append(neighbour)
    return revealed


def get_neighbours(r, c):
    rows, cols = n, n
    return [
        (nr, nc)
        for nr in range(max(0,r-1), min(rows, r+2))
        for nc in range(max(0,c-1), min(cols, c+2))
        if (nr, nc) != (r,c)
    ]

if __name__ == "__main__":
    # play a turn

    choice = random.
    # if this is the first turn, generate the board such that 
    # the choice isn't a GAME_OVER
    # create mine positions
    mine_positions = set(random.sample([
            (a,b) 
            for a in range(n)
            for b in range(n)
            if (a,b) != choice
        ],
        k = mine_count
    ))   
    
    underneath = create_underneath(mine_positions, n)
    print(underneath)

    # update the field
    revealed = reveal(revealed,*choice)
    print(revealed)
    revealed = reveal(revealed,8,8)
    print(revealed)