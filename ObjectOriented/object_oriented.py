# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Marcelo Leszynski
Math 345 Sec 005
12/9/20
"""

import math


class Backpack:
    """A Backpack object class. Has a name, a list of contents, and a 
    maximum size.
    Attributes:
        name (str): the name of the backpack's owner.
        color (str): the color of the backpack.
        contents (list): the contents of the backpack.
        max_size (int): max number of items in backpack.
    """
    ID_index = 0

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.
        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            contents (list): list of items in backpack.
            max_size (int): max number of items in backpack.
        """
        self.name = name
        self.color = color
        self.contents = []
        self.max_size = int(max_size)
        self.ID = self.ID_index 
        self.ID_index +=1

    def put(self, item):
        """
        Add an item to the backpack's list of contents as long as 
        there is still space in the backpack.
        """
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        """Clear all items from the backpack's list of contents."""
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)

    def __eq__(self, other):
        """ Compare backpack names, colors, and contents to check 
        for equality. Returns a boolean. """
        return self.name == other.name and self.color == other.color and len(self.contents) == len(self.contents)

    def __str__(self):
        """ __str__ magic method to enable printing of the 
        Backpack class """
        return "Owner:\t\t" + str(self.name) + "\nColor:\t\t" + str(self.color) + "\nSize:\t\t" + str(len(self.contents)) + "\nMax Size:\t" + str(self.max_size) + "\nContents:\t" + str(self.contents)


# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """
    A jetpack object class that inherits from the Backpack class. Like a 
    backpack, but also stores fuel and flies
    Attributes:
        name (str): the name of the jetpack's owner.
        color (str): the color of the jetpack.
        contents (list): the contents of the jetpack.
        max_size (int): max number of items in jetpack.
        fuel (int): amount of fuel in the jetpack's tank.
    """
    def __init__(self, name, color, max_size = 2, fuel = 10):
        """
        Use the Backpack constructor to initialize a Jetpack object with 
        default max_size of 2 and fuel count of 10.
        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            contents (list): the contents of the jetpack.
            max_size (int): max number of items in jetpack.
            fuel (int): amount of fuel in the jetpack's tank.
        """
        Backpack.__init__(self, name, color, max_size)
        self.fuel = int(fuel)

    def fly(self, fuel):
        """
        consumes fuel to fly as long as input parameter is less than or
        equal to amount of fuel in the jetpack.
        """
        if self.fuel < int(fuel):
            print("Not enough fuel!")
        else:
            self.fuel -= fuel

    def dump(self):
        """Clear all items and fuel from the jetpack."""
        self.contents = []
        self.fuel = 0



# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    """ A complex number class used for computations
    Attributes:
        real (int): the real part of the complex number.
        imag (int): the imaginary coefficient of a complex number.
    """
    def __init__(self, real, imag):
        """
        Initializes an instance of the ComplexNumber class. 
        
        Parameters
            real (int): the real number component of the complex number
            imag (int): the real number scalar of the imaginary component of the complex number
        """
        self.real = int(real)
        self.imag = int(imag)

    def conjugate(self):
        """
        Returns the conjugate of the complex number as a ComplexNumber object.
        """
        return ComplexNumber(self.real, -self.imag)

    def __str__(self):
        """
        Enables formatting for the print() function
        """
        if self.imag >= 0:
            return "(" + str(self.real) + "+" + str(self.imag) + "j)"
        else:
            return "(" + str(self.real) + str(self.imag) + "j)"

    def __abs__(self):
        """
        Calculates the magnitude of the complex number. 
        Returns: (float)
        """
        return math.sqrt(self.real**2 + self.imag**2)

    def __eq__(self, other):
        """
        Checks the real and imaginary parts of 2 complex numbers 
        to check equality. Returns boolean.
        """
        return self.real == other.real and self.imag == other.imag

    def __add__(self, other):
        """
        Add two complex numbers.
        Returns: (ComplexNumber)
        """
        return ComplexNumber(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        """
        Subtract one complex number from another.
        Returns: (ComplexNumber)
        """
        return ComplexNumber(self.real - other.real, self.imag - other.imag)    
        
    def __mul__(self, other):
        """ 
        Multiplies two comples numbers.
        Returns: (ComplexNumber)
        """
        return ComplexNumber(self.real * other.real, self.imag * other.imag)        

    def __truediv__(self, other):
        """
        Computes division of two complex numbers.
        Returns: (ComplexNumber)
        """
        return ComplexNumber(self.real / other.real, self.imag / other.imag)