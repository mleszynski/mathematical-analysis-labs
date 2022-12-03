# standard_library.py
"""Python Essentials: The Standard Library.
Marcelo Leszynski
Math 345 Sec 005
09/08/20
"""
import calculator as calc
import itertools as iter
import sys
import box
import time
import random


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L), max(L), (sum(L) / len(L))


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    print("int, str, tup are immutable, lists and sets are mutable")
    return None


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    return calc.roots(calc.plus(calc.product(a, a), calc.product(b, b)))


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    if len(A) == 0:
        return set(set())
    final_list = []
    for i in range(len(A)+1):
        final_list += set(iter.combinations(A, i))
    return final_list + set()


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    winflag = False
    remaining_nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]                                    # list of remaining numbers
    rolls = 0                                                                       # roll of dice used by players
    start_time = time.time()
    while time.time() < start_time + timelimit:                                     # while there is still time on the clock...
        print("\n")
        p_input = 0
        if sum(remaining_nums) <= 6:                                                # choose whether to roll 1 or 2 dice
            rolls = random.randint(1, 6)
        else:
            rolls = random.randint(2, 12)
        print("Numbers left:", remaining_nums)
        print("Roll:", rolls)
        if not box.isvalid(rolls, remaining_nums):                                  # if it is impossible to win with remaining numbers
            break
        print("Seconds left:", round(start_time + timelimit - time.time(), 2))      # calculate and print the remaining number of seconds
        p_input = input("Numbers to eliminate: ")
        input_lst = box.parse_input(p_input, remaining_nums)
        while len(input_lst) == 0 or not sum(input_lst) == rolls:                   # check for correct player input
            print("Invalid input\n")
            print("Seconds left:", round(start_time + timelimit - time.time(), 2))
            p_input = input("Numbers to eliminate:")                                # ask for reinput if input was invalid
            input_lst = box.parse_input(p_input, remaining_nums)
        for i in range(len(input_lst)):                                             # remove player's chosen numbers from remaining_nums
            remaining_nums.remove(input_lst[i])
        if len(remaining_nums) == 0:                                                # check win condition of no numbers remaining
            winflag = True
    if not winflag:                                                                 # game loss condition
        print("GAME OVER!\n")
    else:                                                                           # game win condition
        print("YOU WON!\n")
    print("Points for player", player + ":", sum(remaining_nums), "points")         # print game info
    print("Time played:", round(time.time() - start_time, 2), "seconds")
    if not winflag:
        print("Better luck next time >:)")
    else:
        print("Congratulations!!!!!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        shut_the_box(sys.argv[1], int(sys.argv[2]))
