# sql1.py
"""Volume 3: SQL 1 (Introduction).
Marcelo Leszynski
Math 347 Sec 003
03/20/21
"""

import sqlite3 as sql
import csv
import numpy as np
from matplotlib import pyplot as plt


# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    # establish connection #####################################################
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

    # drop tables if they exist ################################################
            cur.execute("DROP TABLE IF EXISTS MajorInfo;")
            cur.execute("DROP TABLE IF EXISTS CourseInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentGrades;")

    # create new tables ########################################################
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT);")

    # populate MajorInfo #######################################################
            rows = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", rows)

    # populate CourseInfo ######################################################
            rows = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", rows)

    # populate StudentInfo #####################################################
            with open(student_info, 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentInfo VALUES(?,?,?);", rows)

    # populate StudentGrades ###################################################
            with open(student_grades, 'r') as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO StudentGrades VALUES(?,?,?);", rows)

    # modify StudentInfo according to problem 4 ################################
            cur.execute("UPDATE StudentInfo SET MajorID=NULL WHERE MajorID==-1;")

    # commit or revert, then close connection ##################################
    finally:
        conn.close()


# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    # establish connection #####################################################
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

    # remove old table #########################################################
            cur.execute("DROP TABLE IF EXISTS USEarthquakes;")

    # create new table #########################################################
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, Latitude REAL, Longitude REAL, Magnitude REAL);")

    # populate new table #######################################################
            with open(data_file) as infile:
                rows = list(csv.reader(infile))
                cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)

    # modify table accordint to problem 4 ######################################
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude == 0;")
            cur.execute("UPDATE USEarthquakes SET Day=NULL WHERE Day == 0;")
            cur.execute("UPDATE USEarthquakes SET Hour=NULL WHERE Hour == 0;")
            cur.execute("UPDATE USEarthquakes SET Minute=NULL WHERE Minute == 0;")
            cur.execute("UPDATE USEarthquakes SET Second=NULL WHERE Second == 0;")

    # commit and close connection ##############################################
    finally:
        conn.close()


# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    # establish connection #####################################################
    results = None
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            cur.execute("SELECT SI.StudentName, CI.CourseName "
                        "FROM StudentInfo AS SI, CourseInfo AS CI, StudentGrades as SG "
                        "WHERE (SG.Grade == 'A' OR SG.Grade == 'A+') AND CI.CourseID == SG.CourseID AND SI.StudentID == SG.StudentID;")

            results = cur.fetchall()
    # close connection #########################################################
    finally:
        conn.close()

    return results


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    m_19 = None 
    m_20 = None 
    m_avg = None
    # access database ##########################################################
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

    # read data from table #####################################################
            cur.execute("SELECT Magnitude FROM USEarthquakes "
                        "WHERE USEarthquakes.Year >= 1800 AND USEarthquakes.Year <= 1899;")
            m_19 = np.ravel(cur.fetchall())
            m_19 = np.sort(m_19)

            cur.execute("SELECT Magnitude FROM USEarthquakes "
                        "WHERE USEarthquakes.Year >= 1900 AND USEarthquakes.Year <= 1999;")
            m_20 = np.ravel(cur.fetchall())
            m_20 = np.sort(m_20)

            cur.execute("SELECT AVG(Magnitude) FROM USEarthquakes;")
            m_avg = cur.fetchone()[0]

    finally:
        conn.close()

    # plot histograms ##########################################################
    fig, axs = plt.subplots(2, 1, sharey=True, tight_layout=True)
    axs[0].hist(m_19, 10)
    axs[0].title.set_text('19th Century')
    axs[1].hist(m_20, 10)
    axs[1].title.set_text('20th Century')
    plt.show()

    # return the average #######################################################
    return m_avg
    

#if __name__ == "__main__":
    # test problem 1 ###########################################################
    #student_db()
    #with sql.connect("students.db") as conn:
    #    cur = conn.cursor()
    #    cur.execute("SELECT * FROM StudentInfo;")
    #    print([d[0] for d in cur.description])
    ############################################################################


    # test problem 2 ###########################################################
    #student_db()
    #with sql.connect("students.db") as conn:
    #    cur = conn.cursor()
    #    for row in cur.execute("SELECT * FROM MajorInfo;"):
    #        print(row)
    #    print('\n')
    #    for row in cur.execute("SELECT * FROM CourseInfo;"):
    #        print(row)
    #    print('\n')
    #    for row in cur.execute("SELECT * FROM StudentInfo;"):
    #        print(row)
    #    print('\n')
    #    for row in cur.execute("SELECT * FROM StudentGrades;"):
    #        print(row)
    ############################################################################


    # test problem 3 ###########################################################
    #earthquakes_db()
    #with sql.connect("earthquakes.db") as conn:
    #    cur = conn.cursor()
    #    for row in cur.execute("SELECT * FROM USEarthquakes;"):
    #        print(row)
    ############################################################################


    # test problem 4 part 1 ####################################################
    #student_db()
    #with sql.connect("students.db") as conn:
    #    cur = conn.cursor()
    #    for row in cur.execute("SELECT * FROM StudentInfo;"):
    #        print(row)
    #    print('\n')
    ############################################################################


    # test problem 4 part 2 ####################################################
    #earthquakes_db()
    #with sql.connect("earthquakes.db") as conn:
    #    cur = conn.cursor()
    #    for row in cur.execute("SELECT * FROM USEarthquakes;"):
    #        print(row)
    ############################################################################


    # test problem 5 ###########################################################
    #student_db()
    #print(prob5())
    ############################################################################
    

    # test problem 6 ###########################################################
    #earthquakes_db()
    #print(prob6())
    ############################################################################