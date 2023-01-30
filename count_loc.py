import os
import csv
from pprint import pprint

def search_identical_file(filename, project, program_array):
    for row in program_array:
        if (filename == row[1] and project == row[0]):
            return True
    return False
def count(project_folder, csv_file, language_extension):
    files = os.listdir('.')
    programs = []
    for root, dirs, files in os.walk("Projekte/" + project_folder + "/", topdown=False):
        for file in files:
            name, extension = os.path.splitext(file)
            if (extension == "." + language_extension):
                print(file)
                project = os.path.join(root,file).split("/")[2]
                #print(file)
                #print(os.path.join(root, file))
                with open(os.path.join(root, file), 'r') as source:
                    num_lines = len(source.readlines())
                if (search_identical_file(file, project, programs)):
                    parent_dir = os.path.basename(root)
                    file = name + "_" + parent_dir + extension
                    print(file)
                programs.append([project, file, num_lines])

    #pprint(programs)

    with open(csv_file, "w", newline='', encoding='utf-8') as count_programs:
        writer = csv.writer(count_programs)
        writer.writerow(["Project", "Case ID","LOC"])
        for program in programs:
            writer.writerow(program)


count("Python", "python_programs.csv", "py")
count("Java", "java_programs.csv", "java")

