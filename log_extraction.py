import ast
import astor
import astpretty
from pprint import pprint
import os
import csv
import javalang
import logging

# count loc

log_levels = ["info", "error", "log", "severe", "debug", "warn", "trace", "quiet", "critical", "warning", "fine"]

def search_identical_file(filename, project, program_array):
    for row in program_array:
        if (filename == row[0] and project == row[2]):
            return True
    return False

pyprograms = []
files = os.listdir('.')
for root, dirs, files in os.walk("Projekte/Python/", topdown=False):
    for file in files:
        name, extension = os.path.splitext(file)
        if (extension == ".py"):
            project = os.path.join(root, file).split("/")[2]
            #print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r') as source:
                if (search_identical_file(file, project, pyprograms)):
                    parent_dir = os.path.basename(root)
                    file = parent_dir + "/" + name  + extension
                pyprograms.append([file, source.read(), project])

programs = []
files = os.listdir('.')
for root, dirs, files in os.walk("Projekte/Java/", topdown=False):
    for file in files:
        name, extension = os.path.splitext(file)
        if(extension == ".java"):
            project = os.path.join(root, file).split("/")[2]
            #print(os.path.join(root, file))
            with open(os.path.join(root, file), 'r') as source:
                if (search_identical_file(file, project, programs)):
                    parent_dir = os.path.basename(root)
                    file = parent_dir + "/" + name  + extension
                programs.append([file, source.read(), project])

def write_to_file_python(log_array):
    with open("python_event_log.csv", "a", newline='', encoding='utf-8') as python_event_log:
        writer = csv.writer(python_event_log, delimiter="|")
        for element in log_array:
            data = []
            if (str(element[4]).lower() == "warning"):
                element[4] = "warn"
            if (len(element) > 6):
                # print(element)
                if (element[3] < 100):
                    element[3] = "0" + str(element[3])
                data = [element[1], str(element[3]), element[2], str(element[4])]
                if (isinstance(element[5], str) and "=>" in element[5]):
                    index = element[5].find("=")
                    element[5] = element[5][:index] + "\\" + element[5][index:]
                elif (isinstance(element[5], str) and "=========" in element[5]):
                    # item = item.replace("=", "-")
                    index = element[5].find("====")
                    element[5] = element[5][:index] + "\\" + element[5][index:]
                log_statement = str(element[5]) + ", "

                for item in element[6:]:
                    log_statement += str(item) + ", "
                data.append(log_statement[:-2])
                data.append(element[0])
            elif (len(element) == 6):
                if (isinstance(element[5], str) and "====" in element[5]):
                    element[5] = element[5].replace("=", "-")
                data = [element[1], str(element[3]), element[2], str(element[4]), str(element[5]), element[0]]
            else:  # empty log statement
                data = [element[1], str(element[3]), element[2], str(element[4]), "", element[0]]
            writer.writerow(data)

def write_to_file_java(log_array):
    with open("java_event_log.csv", "a") as java_event_log:
        writer = csv.writer(java_event_log, delimiter="|")
        for element in log_array:
            if (str(element[4]).lower() == "warning"):
                element[4] = "warn"
            data = []
            #Case ID", "Start Timestamp", "Resource", "Role", "Activity
            data = [element[1],str(element[2]),element[3] ,str(element[4]).lower(), None, None]
            if (len(element) >= 6):
                log_text = element[5] + " "
                for statement in element[6:]:
                    log_text += statement + " "
                data[4] = log_text
            data[5] = element[0]
            writer.writerow(data)

# ast library seemingly has trouble detecting nested if-statements so I'm checking other blocks manually for them
def traverse_if(if_node):
    for el in if_node.body:
        if (isinstance(el, ast.IfExp) or isinstance(el, ast.If)):
            traverse_if(el)
        if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                el.value.func.value, "id") and (
                el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
            FuncListener().if_logging.append([projectname, filename, "If Statement", el.lineno, el.value.func.attr])
            if (hasattr(el.value, "args")):
                for arg in el.value.args:
                    traverse_LogStatement_ast(arg, FuncListener().if_logging)
    if (hasattr(if_node, "orelse")):
        for el in if_node.orelse:
            if (isinstance(el, ast.IfExp) or isinstance(el, ast.If)):
                traverse_if(el)
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().if_logging.append(
                    [projectname, filename, "If Statement", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().if_logging)


# traversing all node of log statement to save both text and variable names
def traverse_LogStatement_ast(statement_node, log_array):
    if (isinstance(statement_node, ast.BinOp)):
        traverse_LogStatement_ast(statement_node.left, log_array)
        traverse_LogStatement_ast(statement_node.right, log_array)
    if (isinstance(statement_node, ast.Call)):
        traverse_LogStatement_ast(statement_node.func, log_array)
        if (hasattr(statement_node, "args")):
            for arg in statement_node.args:
                traverse_LogStatement_ast(arg, log_array)
        if (hasattr(statement_node, "func")):
            traverse_LogStatement_ast(statement_node.func, log_array)
    if (isinstance(statement_node, ast.Attribute)):
        traverse_LogStatement_ast(statement_node.value, log_array)
    if (isinstance(statement_node, ast.Name)):
        log_array[-1].append(statement_node.id)
    if (isinstance(statement_node, ast.Constant)):
        log_array[-1].append(statement_node.value)
    if (isinstance(statement_node, ast.Subscript)):
        if (hasattr(statement_node.value, "id")):
            log_array[-1].append(statement_node.value.id)
            traverse_LogStatement_ast(statement_node.slice, log_array)
            # print(statement_node.__dict__)
            # print(statement_node.value.id)
        elif (hasattr(statement_node.value, "func")):
            traverse_LogStatement_ast(statement_node.value, log_array)

def traverse_LogStatement_javalang(node):
    if(isinstance(node, javalang.tree.Literal)):
        logging_statements[-1].append(node.value)
    if(isinstance(node, javalang.tree.BinaryOperation)):
        if(hasattr(node, "operandl")):
            traverse_LogStatement_javalang(node.operandl)
        if (hasattr(node, "operandr")):
            traverse_LogStatement_javalang(node.operandr)
    if (isinstance(node, javalang.tree.MemberReference)):
        logging_statements[-1].append(node.member)
    if (isinstance(node, javalang.tree.MethodInvocation)):
        function_call = str(node.qualifier)
        if(hasattr(node, "member") and not isinstance(node.member, type(None))):
            function_call += "." + str(node.member)
        logging_statements[-1].append(function_call)

class FuncListener(ast.NodeVisitor):
    function_logging = []
    if_logging = []
    Try_logging = []
    Match_logging = []
    For_logging = []
    While_logging = []
    With_logging = []

    def visit_FunctionDef(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        # Log Statement with formatting
                        if (hasattr(arg, "func") and hasattr(arg.func, "value") and isinstance(arg.func.value,ast.Constant)):
                            FuncListener().function_logging.append(
                                [projectname, filename, "Method", el.lineno, el.value.func.attr, arg.func.value.value])
                            if (hasattr(arg, "args")):
                                for arg in arg.args:
                                    if (hasattr(arg, "id")):  # Variable in Log Statement
                                        FuncListener().function_logging[-1].append(arg.id)
                        elif (hasattr(arg, "id")):  # only Variable
                            FuncListener().function_logging.append(
                                [projectname, filename, "Method", el.lineno, el.value.func.attr, arg.id])
                        elif (hasattr(arg, "value")):  # Log Statement pure
                            FuncListener().function_logging.append(
                                [projectname, filename, "Method", el.lineno, el.value.func.attr, arg.value])
        self.generic_visit(node)
        return node

    def visit_If(self, node):
        if_obj = node
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().if_logging.append([projectname, filename, "If Statement", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().if_logging)

        for el in node.orelse:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().if_logging.append([projectname, filename, "If Statement", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().if_logging)
            if (isinstance(el, ast.If)):
                for el in el.body:
                    if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func,"value") and hasattr(el.value.func.value, "id") and (el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                        FuncListener().if_logging.append(
                            [projectname, filename, "If Statement", el.lineno, el.value.func.attr])
                        if (hasattr(el.value, "args")):
                            for arg in el.value.args:
                                traverse_LogStatement_ast(arg, FuncListener().if_logging)
                if (hasattr(el, "orelse")):
                    for el in el.orelse:
                        if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func,
                                                                                               "value") and hasattr(
                                el.value.func.value, "id") and (
                                el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                            FuncListener().if_logging.append(
                                [projectname, filename, "If Statement", el.lineno, el.value.func.attr])
                            if (hasattr(el.value, "args")):
                                for arg in el.value.args:
                                    traverse_LogStatement_ast(arg, FuncListener().if_logging)

        self.generic_visit(node)

    def visit_Try(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().Try_logging.append([projectname, filename, "Try", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().Try_logging)
        for el in node.orelse:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().Try_logging.append([projectname, filename, "Try Else", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().Try_logging)
        for el in node.finalbody:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().Try_logging.append([projectname, filename, "Try Finally", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().Try_logging)
        for handler in node.handlers:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().Try_logging.append([projectname, filename, "Exception", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().Try_logging)

    """ no match statement in Python 3.8 """

    def visit_Match(self, node):
        for case in node.cases:
            for el in case.body:
                if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(el.value.func.value, "id") and (el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                    FuncListener().Match_logging.append(
                        [projectname, filename, "match case", el.lineno, el.value.func.attr])
                    if (hasattr(el.value, "args")):
                        for arg in el.value.args:
                            traverse_LogStatement_ast(arg, FuncListener().Match_logging)

    def visit_For(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().For_logging.append([projectname, filename, "For", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().For_logging)

    def visit_While(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().While_logging.append([projectname, filename, "While", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().While_logging)
            if (isinstance(el, ast.If) or isinstance(el, ast.IfExp)):
                traverse_if(el)
        for el in node.orelse:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().While_logging.append([projectname, filename, "While", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().While_logging)
            if (isinstance(el, ast.If) or isinstance(el, ast.IfExp)):
                traverse_if(el)

    def visit_With(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().With_logging.append([projectname, filename, "With", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().With_logging)

    def visit_AsyncFunctionDef(self, node):
        for el in node.body:
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().function_logging.append([projectname, filename, "Method", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().function_logging)
        self.generic_visit(node)
        return node

    def visit_AsyncWith(self, node):
        for el in node.body:
            print("ASYNCH WITH!")
            print(el)
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().With_logging.append([projectname, filename, "With", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().With_logging)
        self.generic_visit(node)
        return node

    def visit_AsyncFor(self, node):
        for el in node.body:
            print("asynch for!")
            print(el)
            if (isinstance(el, ast.Expr) and hasattr(el.value, "func") and hasattr(el.value.func, "value") and hasattr(
                    el.value.func.value, "id") and (
                    el.value.func.value.id == "logger" or el.value.func.value.id == "logging") and el.value.func.attr in log_levels):
                FuncListener().For_logging.append([projectname, filename, "For", el.lineno, el.value.func.attr])
                if (hasattr(el.value, "args")):
                    for arg in el.value.args:
                        traverse_LogStatement_ast(arg, FuncListener().For_logging)
        self.generic_visit(node)
        return node






# extract logging code from Python projects



# filter logging code from each python program file
for program in pyprograms:
    filename = program[0]
    projectname = program[2]
    print(filename)
    FuncListener().visit(ast.parse(program[1]))

with open("python_event_log.csv", "w", newline='', encoding='utf-8') as python_event_log:
    writer = csv.writer(python_event_log, delimiter="|")
    writer.writerow(["Case ID", "Start Timestamp",
                     "Resource", "Role", "Activity", "Project"])
write_to_file_python(FuncListener().function_logging)

write_to_file_python(FuncListener().if_logging)
write_to_file_python(FuncListener().Match_logging)
write_to_file_python(FuncListener().While_logging)
write_to_file_python(FuncListener().With_logging)
write_to_file_python(FuncListener().For_logging)
write_to_file_python(FuncListener().Try_logging)

print("python complete")

LOG = "exceptions.log"
logging.basicConfig(filename=LOG, filemode="w", level=logging.INFO)

# extract logging code from Java projects



logging_statements = []

for program in programs:
    project_name = program[2]
    try:
        tree = javalang.parse.parse(program[1])
        file_name = program[0]
        print(file_name)
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            if(hasattr(node, "body") and not isinstance(node.body, type(None))):
                for subnode in node.body:
                    if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                        #print(subnode.expression.arguments[0])
                        if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append( [project_name, file_name, subnode.position.line, "Method", subnode.expression.arguments[0].member])
                            for arg in subnode.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not subnode.expression.member in log_levels):
                            if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "Method",  subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, subnode.position.line, "Method", "log"])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append([project_name, file_name, subnode.position.line, "Method", subnode.expression.member])
                            for arg in subnode.expression.arguments:
                                traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.IfStatement):
            if (hasattr(node, "then_statement") and not isinstance(node.then_statement, type(None)) and not isinstance(node.then_statement, javalang.tree.ContinueStatement)):
                if(isinstance(node.then_statement, javalang.tree.BlockStatement)):
                    for subnode in node.then_statement.statements:
                        if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression,javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                            if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                                logging_statements.append( [project_name, file_name, subnode.position.line, "If Statement", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments[1:]:
                                    traverse_LogStatement_javalang(arg)
                            elif (not subnode.expression.member in log_levels):
                                if(subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                    logging_statements.append([project_name, file_name, subnode.position.line, "If Statement", subnode.expression.arguments[0].member])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                                else:
                                    # log level either not given or unable to find it
                                    logging_statements.append([project_name, file_name, subnode.position.line, "If Statement", "log"])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                            else:
                                logging_statements.append([project_name, file_name, subnode.position.line, "If Statement", subnode.expression.member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                else:
                    if (isinstance(node.then_statement, javalang.tree.StatementExpression) and isinstance(node.then_statement.expression, javalang.tree.MethodInvocation) and (node.then_statement.expression.qualifier.casefold() == "logger" or node.then_statement.expression.qualifier.casefold() == "logging" or node.then_statement.expression.qualifier.casefold() == "log")):
                        if (node.then_statement.expression.member == "log" and hasattr(node.then_statement.expression.arguments[0], "qualifier") and node.then_statement.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append([project_name, file_name, node.then_statement.position.line, "If Statement",node.then_statement.expression.arguments[0].member])
                            for arg in node.then_statement.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not node.then_statement.expression.member in log_levels):
                            if (node.then_statement.expression.arguments and hasattr(node.then_statement.expression.arguments[0], "qualifier") and node.then_statement.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, node.then_statement.position.line, "If Statement",  node.then_statement.expression.arguments[0].member])
                                for arg in node.then_statement.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, node.then_statement.position.line, "If Statement", "log"])
                                for arg in node.then_statement.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append([project_name, file_name, node.then_statement.position.line, "If Statement", node.then_statement.expression.member])
                            for arg in node.then_statement.expression.arguments:
                                traverse_LogStatement_javalang(arg)



        for path, node in tree.filter(javalang.tree.WhileStatement):
            if (hasattr(node, "body") and not isinstance(node.body, type(None))):
                if (isinstance(node.body, javalang.tree.BlockStatement)):
                    for subnode in node.body.statements:
                        if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                            if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                                logging_statements.append( [project_name, file_name, subnode.position.line, "While", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments[1:]:
                                    traverse_LogStatement_javalang(arg)
                            elif (not subnode.expression.member in log_levels):
                                if(subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                    logging_statements.append([project_name, file_name, subnode.position.line, "While", subnode.expression.arguments[0].member])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                                else:
                                    # log level either not given or unable to find it
                                    logging_statements.append([project_name, file_name, subnode.position.line, "While", "log"])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                            else:
                                logging_statements.append( [project_name, file_name, subnode.position.line, "While", subnode.expression.member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                else:
                    stmnt = node.body
                    if (isinstance(stmnt, javalang.tree.StatementExpression) and isinstance(stmnt.expression, javalang.tree.MethodInvocation) and (stmnt.expression.qualifier.casefold() == "logger" or stmnt.expression.qualifier.casefold() == "logging" or stmnt.expression.qualifier.casefold() == "log")):
                        if (stmnt.expression.member == "log" and hasattr(stmnt.expression.arguments[0], "qualifier") and stmnt.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append([project_name, file_name, stmnt.position.line, "If Statement", stmnt.expression.arguments[0].member])
                            for arg in stmnt.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not stmnt.expression.member in log_levels):
                            if(stmnt.expression.arguments and hasattr(stmnt.expression.arguments[0], "qualifier") and stmnt.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, stmnt.position.line, "If Statement", stmnt.expression.arguments[0].member])
                                for arg in stmnt.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, stmnt.position.line, "If Statement", "log"])
                                for arg in stmnt.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append([project_name, file_name, stmnt.position.line, "If Statement", stmnt.expression.member])
                            for arg in stmnt.expression.arguments:
                                traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.DoStatement):
            if (hasattr(node, "body") and not isinstance(node.body, type(None))):
                for subnode in node.body.statements:
                    if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                        if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append( [project_name, file_name, subnode.position.line, "While", subnode.expression.arguments[0].member])
                            for arg in subnode.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not subnode.expression.member in log_levels):
                            if(subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "While", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, subnode.position.line, "While", "log"])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append( [project_name, file_name, subnode.position.line, "While", subnode.expression.member])
                            for arg in subnode.expression.arguments:
                                traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.ForStatement):
            if (hasattr(node, "body") and not isinstance(node.body, type(None))):
                if(isinstance(node.body, javalang.tree.BlockStatement)):
                    for subnode in node.body.statements:
                        if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                            if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                                logging_statements.append( [project_name, file_name, subnode.position.line, "For", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments[1:]:
                                    traverse_LogStatement_javalang(arg)
                            elif (not subnode.expression.member in log_levels):
                                if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                    logging_statements.append([project_name, file_name, subnode.position.line, "For",subnode.expression.arguments[0].member])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                                else:
                                    # log level either not given or unable to find it
                                    logging_statements.append([project_name, file_name, subnode.position.line, "For", "log"])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                            else:
                                logging_statements.append( [project_name, file_name, subnode.position.line, "For", subnode.expression.member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                else:
                    stmnt = node.body
                    if (isinstance(stmnt, javalang.tree.StatementExpression) and isinstance(stmnt.expression, javalang.tree.MethodInvocation) and (stmnt.expression.qualifier.casefold() == "logger" or stmnt.expression.qualifier.casefold() == "logging" or stmnt.expression.qualifier.casefold() == "log")):
                        if (stmnt.expression.member == "log" and hasattr(stmnt.expression.arguments[0], "qualifier") and stmnt.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append([project_name, file_name, stmnt.position.line, "For", stmnt.expression.arguments[0].member])
                            for arg in stmnt.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not stmnt.expression.member in log_levels):
                            if(hasattr(stmnt.expression.arguments and stmnt.expression.arguments[0], "qualifier") and stmnt.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, stmnt.position.line, "For", stmnt.expression.arguments[0].member])
                                for arg in stmnt.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, stmnt.position.line, "For", "log"])
                                for arg in stmnt.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append([project_name, file_name, stmnt.position.line, "For", stmnt.expression.member])
                            for arg in stmnt.expression.arguments:
                                traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.SynchronizedStatement):
            if (hasattr(node, "block") and not isinstance(node.block, type(None))):
                for subnode in node.block:
                    if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                        if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append( [project_name, file_name, subnode.position.line, "Synchronized", subnode.expression.arguments[0].member])
                            for arg in subnode.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not subnode.expression.member in log_levels):
                            if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "Synchronized",subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, subnode.position.line, "Synchronized", "log"])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append( [project_name, file_name, subnode.position.line, "Synchronized", subnode.expression.member])
                            for arg in subnode.expression.arguments:
                                traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.TryStatement):
            if (hasattr(node, "block") and not isinstance(node.block, type(None))):
                for subnode in node.block:
                    if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                        if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append( [project_name, file_name, subnode.position.line, "Try", subnode.expression.arguments[0].member])
                            for arg in subnode.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not subnode.expression.member in log_levels):
                            if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "Try",subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, subnode.position.line, "Try", "log"])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append( [project_name, file_name, subnode.position.line, "Try", subnode.expression.member])
                            for arg in subnode.expression.arguments:
                                traverse_LogStatement_javalang(arg)

        for path, node in tree.filter(javalang.tree.SwitchStatement):
            if (hasattr(node, "cases") and not isinstance(node.cases, type(None))):
                for case in node.cases:
                    for subnode in case.statements:
                        if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                            if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "Switch", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments[1:]:
                                    traverse_LogStatement_javalang(arg)
                            elif (not subnode.expression.member in log_levels):
                                if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                    logging_statements.append([project_name, file_name, subnode.position.line, "Switch", subnode.expression.arguments[0].member])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                                else:
                                    # log level either not given or unable to find it
                                    logging_statements.append([project_name, file_name, subnode.position.line, "Switch", "log"])
                                    for arg in subnode.expression.arguments:
                                        traverse_LogStatement_javalang(arg)
                            else:
                                logging_statements.append( [project_name, file_name, subnode.position.line, "Switch", subnode.expression.member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)


        for path, node in tree.filter(javalang.tree.CatchClause):
            if (hasattr(node, "block") and not isinstance(node.block, type(None))):
                for subnode in node.block:
                    if (isinstance(subnode, javalang.tree.StatementExpression) and isinstance(subnode.expression, javalang.tree.MethodInvocation) and (subnode.expression.qualifier.casefold() == "logger" or subnode.expression.qualifier.casefold() == "logging" or subnode.expression.qualifier.casefold() == "log")):
                        if (subnode.expression.member == "log" and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Level"):
                            logging_statements.append([project_name, file_name, subnode.position.line, "Exception", subnode.expression.arguments[0].member])
                            for arg in subnode.expression.arguments[1:]:
                                traverse_LogStatement_javalang(arg)
                        elif (not subnode.expression.member in log_levels):
                            if (subnode.expression.arguments and hasattr(subnode.expression.arguments[0], "qualifier") and subnode.expression.arguments[0].qualifier == "Kind"):
                                logging_statements.append([project_name, file_name, subnode.position.line, "Exception", subnode.expression.arguments[0].member])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                            else:
                                # log level either not given or unable to find it
                                logging_statements.append([project_name, file_name, subnode.position.line, "Exception", "log"])
                                for arg in subnode.expression.arguments:
                                    traverse_LogStatement_javalang(arg)
                        else:
                            logging_statements.append([project_name, file_name, subnode.position.line, "Exception", subnode.expression.member])
                            for arg in subnode.expression.arguments:
                                traverse_LogStatement_javalang(arg)

    except javalang.parser.JavaSyntaxError as err:
        logging.info(program[0])
        logging.info("JavaSyntaxError")

    except javalang.tokenizer.LexerError as err:
        logging.info(program[0])
        logging.info("LexerError ")


with open("java_event_log.csv", "w") as java_event_log:
    writer = csv.writer(java_event_log, delimiter="|")
    writer.writerow(["Case ID", "Start Timestamp", "Resource", "Role", "Activity", "Project"])

write_to_file_java(logging_statements)

