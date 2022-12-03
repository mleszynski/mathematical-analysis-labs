# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Marcelo Leszynski
Math 345 Sec 005
09/24/20
"""

from random import choice
import numpy as np


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:

    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """

    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:  # user input is of incorrect length
        raise ValueError("Input should be a three digit number")
    if abs(int(step_1[0]) - int(step_1[2])) < 2:  # first and last user input don't differ by at least 2
        raise ValueError("First and last digits sshould differ by at least 2")
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    if step_2 != step_1[:, -1]:  # if user input isn't reverse of first input
        raise ValueError("The second input should be the reverse of the first input")
    step_3 = input("Enter the positive difference of these numbers: ")
    if step_3 != abs(int(step_1) - int(step_2)):  # if user input is incorrect
        raise ValueError("The third input should be the positive difference of the first two inputs")
    step_4 = input("Enter the reverse of the previous result: ")
    if step_4 != step_3[:, -1]:  # step 4 isn't the reverse of step 3
        raise ValueError("The fourth input should be the reverse of the third input")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the
    program is running, the function should catch the exception and
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    try:
        walk = 0
        directions = [1, -1]
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("\nProcess interrupted at iteration ", i)
    else:
        print("Process completed")
    finally:
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file

    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
    """
class ContentFilter(object):
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        while True:
            try:
                myfile = open(filename)  # open file
                linelist = myfile.readlines()  # read in text from file
                self.contents = ""
                for i in range(len(linelist)):  # store text in object
                    self.contents += str(linelist[i])
                break
            except FileNotFoundError:
                filename = input("Please enter a valid file name: ")
            except TypeError:
                filename = input("Please enter a valid file name: ")
            except OSError:
                filename = input("Please enter a valid file name: ")
        myfile.close()  # close input file
        self.filename = filename  # store the name of input file in object
        self.total_chars = len(self.contents)  # store num chars in file
        self.alpha_chars = sum([s.isalpha() for s in self.contents])  # store num letters in file
        self.numer_chars = sum([s.isdigit() for s in self.contents])  # store num numbers in file
        self.white_chars = sum([s.isspace() for s in self.contents])  # store num whitespaces in file
        self.num_lines = 0
        for char in self.contents:  # count num newlines in file
            if char == "\n":
                self.num_lines += 1

# Problem 4 ---------------------------------------------------------------

    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if not (mode == 'w' or mode == 'x' or mode == 'a'):
            raise ValueError("Incorrect mode input!")

    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data ot the outfile in uniform case."""
        self.check_mode(mode)
        with open(outfile, mode) as outy:
            if case == "upper":
                outy.write(self.contents.upper())
            elif case == "lower":
                outy.write(self.contents.lower())
            else:
                raise ValueError("Incorrect case input!")


    def reverse(self, outfile, mode='w', unit='word'):
        """Write the data to the outfile in reverse order."""
        self.check_mode(mode)
        with open(outfile, mode) as outy:
            if unit == "word":
                linesplit = self.contents.split('\n')  # split text across lines
                linesplit.pop()
                for line in linesplit:  # split text across words
                    wordsplit = line.split(' ')
                    outy.write(' '.join(wordsplit[::-1]))
                    outy.write('\n')
            elif unit == "line":  # split and reverse across lines
                output = self.contents.split('\n')
                output.pop()
                outy.write('\n'.join(output[::-1]))
            else:
                raise ValueError("Incorrect unit input!")

    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        with open(outfile, mode) as outy:
            contents = self.contents.split('\n')
            contents.pop()
            words = []
            for i in contents:  # create list of words
                words.append(i.split(' '))
            tarray = np.array(words)  # turn words list into array
            tarray = np.transpose(tarray)  # transpose word array
            for row in tarray[:, ]:
                outy.write(' '.join(row))
                outy.write('\n')

    def __str__(self):
        """String representation: info about the contents of the file."""
        output = "Source file:\t\t\t" + str(self.filename) + '\n'
        output += "Total characters:\t\t" + str(self.total_chars) + '\n'
        output += "Alphabetic characters:\t\t" + str(self.alpha_chars) + '\n'
        output += "Numerical characters:\t\t" + str(self.numer_chars) + '\n'
        output += "Whitespace characters:\t\t" + str(self.white_chars) + '\n'
        output += "Number of lines:\t\t" + str(self.num_lines) + '\n'
        return output
