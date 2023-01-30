# This needs to be rewritten in OOP format... or DOES it?!?!?!
import datetime
import logging
import queue
import sys
import threading
import time
from sys import exit
from matplotlib import ticker, dates
import matplotlib.pyplot as plt
import serial
import serial.tools.list_ports
import os.path
import csv
from remotecommunication import get_credentials, send_mass_email_with_log, send_mass_sms, user_setup

# global queues used to house stings from/to serial
validation_queue = queue.Queue()
data_queue = queue.Queue()
input_queue = queue.Queue()

# globals for graphing
plt.ion()
fig, (ax3, ax2, ax1) = plt.subplots(3)
historic_date_times = []
historic_co2_ppm = []
historic_temps = []
historic_feed_rotations = []

# global flags
data_updated = False
exit_program = False
reset_program = False
pause_input = False
stop_input = False


def start_logger():
    home = os.path.expanduser('~')
    log_file = os.path.join(home, '.bioreactor')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file = os.path.join(log_file, '.log')
    if not os.path.exists(log_file):
        os.mkdir(log_file)
    log_file = os.path.join(log_file, 'log.log')
    FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s [%(funcName)s] %(message)s'
    logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO, filename=log_file)
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(levelname)s [%(funcName)s] %(message)s')
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    return logger


def all_points_bulletin(logger, subject, message):
    service = get_credentials()
    send_mass_sms(service, subject, message)
    send_mass_email_with_log(service, message)
    logger.info('An all points bulletin was sent out with message: {0}'.format(message))


def limit_list_size(array, limit_size):
    if len(array) > limit_size:
        slice_end = len(array) - limit_size
        del array[0:slice_end]
    return array


def write_serial_data(serial_connection, string):
    string = string + '\n'
    string = bytes(string, 'utf-8')
    serial_connection.write(string)


def shake_hand(serial_connection, string, logger):
    if string == "FLASH":
        response = 'THUNDER'
        write_serial_data(serial_connection, response)
        logger.info('Responded to handshake request')
        get_validation(serial_connection, logger)
        print('Type `help` for command information')


def validate(serial_connection, check, logger):
    if check:
        string = 'CONFIRMED'
        write_serial_data(serial_connection, string)
        logger.info('Sent confirming validation')
    else:
        string = 'FAILED'
        write_serial_data(serial_connection, string)
        logger.error('Sent failing validation')


def get_validation(serial_connection, logger):
    timeout = time.time() + 60  # 60 second timeout for function to complete
    while time.time() < timeout:
        if serial_connection.in_waiting > 0:
            line = serial_connection.readline().decode('utf-8').rstrip()
            if line == 'CONFIRMED':
                logger.info('Got affirming validation')
                return True
            elif line == 'FAILED':
                logger.info('Got failing validation')
                return False
            else:
                # print('ERROR: Validation Failed!')
                logger.error('Validation answer not understood')
                return False
    error_string = 'Validation request timed out. Check microcontroller'
    logger.error(error_string)
    all_points_bulletin(logger, 'ERROR', error_string)
    return False


def read_serial_data(serial_connection, logger):
    global data_queue, validation_queue, pause_input
    while serial_connection.in_waiting > 0:
        # strip serial string of unnecessary garbage
        line = serial_connection.readline().decode('utf-8').rstrip()
        if line == "FLASH":
            logger.info('Handshake request received')
            shake_hand(serial_connection, line, logger)
        elif line == "READING":
            pause_input = True
            logger.info('Read/Feed operation initiated')
            validate(serial_connection, True, logger)
            print('Please wait for the read/feed operation to resolve...')
        elif line == "READDONE":
            pause_input = False
            logger.info('Feed action complete')
            validate(serial_connection, True, logger)
        else:
            values = line.split(",")
            if values[0] == 'DATA' or values[0] == 'DATAFEED':
                data_queue.put(line)
                logger.info('Data received: ' + line)
                validate(serial_connection, True, logger)
            elif values[0] == 'SUCCESS' or values[0] == 'ERROR':
                validation_queue.put(line)
                logger.info('Message from microcontroller: ' + line)
                validate(serial_connection, True, logger)
            elif values[0] == 'CURFEEDPPM':
                logger.info('Current feed CO2 threshold: ' + values[1])
                validate(serial_connection, True, logger)
            elif values[0] == 'CURFEEDROT':
                logger.info('Current feed rotations: ' + values[1])
                validate(serial_connection, True, logger)
            else:
                # print('ERROR: Data could not be parsed from the serial buffer!')
                error_string = 'Data could not be parsed from the serial buffer: ' + line
                logger.error(error_string)
                validate(serial_connection, False, logger)
                all_points_bulletin(logger, 'ERROR', error_string)
    return


def print_validations(logger):
    global validation_queue
    while not validation_queue.empty():
        line = validation_queue.get()
        values = line.split(",")
        validation = values[0]
        message = values[1]
        # print('FROM CONTROLLER:' + validation + ':' + message)
        logger.info(validation + ' from controller: ' + message)
        validation_queue.task_done()
        if validation == 'ERROR':
            all_points_bulletin(logger, validation, message)
    return


def log_data(read_time, ppm, temp, feed_rotations, logger):
    read_time = datetime.datetime.strftime(read_time, '%Y-%m-%d %H:%M:%S')
    headers = ['date_time_stamp', 'relative_ppm', 'temp', 'feed_rotations']
    home_dir = os.path.expanduser('~')
    csv_path = os.path.join(home_dir, '.bioreactor')
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    csv_path = os.path.join(csv_path, '.log')
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    csv_path = os.path.join(csv_path, 'data.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            writer.writeheader()  # file doesn't exist yet, write a header
    else:
        with open(csv_path, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
            writer.writerow(
                {'date_time_stamp': read_time, 'relative_ppm': ppm, 'temp': temp, 'feed_rotations': feed_rotations})
    csvfile.close()
    logger.info('Data logged')
    return


def store_data(logger):
    global data_queue, historic_date_times, historic_co2_ppm, historic_temps, historic_feed_rotations, data_updated
    list_size_limit = 200
    # read the serial, parse out data and cast as appropriate types
    while not data_queue.empty():
        read_time = datetime.datetime.now()
        line = data_queue.get()
        values = line.split(",")
        if values[0] == 'DATA':
            ppm_raw = values[1]
            temp_raw = values[2]
            ppm = int(ppm_raw)
            temp = float(temp_raw)
            feed_rotations = 0
        elif values[0] == 'DATAFEED':
            ppm_raw = values[1]
            temp_raw = values[2]
            feed_rotations_raw = values[3]
            ppm = int(ppm_raw)
            temp = float(temp_raw)
            feed_rotations = int(feed_rotations_raw)
        else:
            # print('ERROR: Data grab failed!')
            error_string = 'Data grab failed'
            logger.error(error_string)
            all_points_bulletin(logger, 'ERROR', error_string)
            return
        # throw error if CO2 is out of range
        if not ppm >= 100 or ppm > 10000:
            # print('ERROR: CO2 concentration is out of range!')
            error_string = 'CO2 concentration is out of range'
            logger.error(error_string)
            all_points_bulletin(logger, 'ERROR', error_string)
            return
        else:
            historic_date_times.append(read_time)
            historic_date_times = limit_list_size(historic_date_times, list_size_limit)
            historic_co2_ppm.append(ppm)
            historic_co2_ppm = limit_list_size(historic_co2_ppm, list_size_limit)
            historic_temps.append(temp)
            historic_temps = limit_list_size(historic_temps, list_size_limit)
            historic_feed_rotations.append(feed_rotations)
            historic_feed_rotations = limit_list_size(historic_feed_rotations, list_size_limit)
            log_data(read_time, ppm, temp, feed_rotations, logger)
            data_queue.task_done()
            logger.info('Data was added to lists')
            data_updated = True
            return
    return


def handle_data(serial_connection, logger):
    read_serial_data(serial_connection, logger)
    print_validations(logger)
    store_data(logger)


def queue_keyboard_input():
    global input_queue, stop_input, pause_input, exit_program
    if not exit_program:
        while True:
            keyboard_input = sys.stdin.readline()
            if stop_input:
                break
            elif pause_input:
                print('Command ignored. Please wait for read event to resolve')
            elif keyboard_input != '\n':
                input_queue.put(keyboard_input)


# command prompt function
def command_line_interface(serial_connection, logger):
    global reset_program, exit_program, input_queue, stop_input
    if not input_queue.empty():
        command_raw = input_queue.get()
        command_raw = command_raw.rstrip('\n')
        if command_raw == 'help':
            print(
                "Available commands:\n   reset [INT VALUE] (1 for controller, 0 for program)\n   "
                "get-feed-rotations\n    get-feed-ppm\n"
                "set-feed-rotations [INT VALUE] (1-100)\n   set-feed-ppm [INT VALUE] (500-5000)\n   exit ("
                "exits program)\nFormat: [COMMAND] OPTIONAL[VALUE]\nExample: reset 1 (where 1 resets the controller)\n")
        elif command_raw == 'get-feed-rotations':
            logger.info('Requesting current feed rotations. Please wait...')
            write_serial_data(serial_connection, command_raw)
            get_validation(serial_connection, logger)

        elif command_raw == 'get-feed-ppm':
            logger.info('Requesting current feed CO2 ppm threshold. Please wait...')
            write_serial_data(serial_connection, command_raw)
            get_validation(serial_connection, logger)
        elif command_raw == 'exit':
            logger.info('Initiating exit')
            # print
            stop_input = True
            exit_program = True
            print('Press ENTER to exit program')
        else:
            command_separated = command_raw.split(' ')
            try:
                logger.info('Attempting command ' + command_separated[0] + ' with a value of '
                            + command_separated[1])
            except IndexError:
                print('Unrecognized command: ' + command_raw)
                return
            # print('Attempting command', command_separated[0], 'with a value of', command_separated[1])
            command = command_separated[0]
            try:
                value = int(command_separated[1])
            except ValueError:
                print('Command [VALUE] must be an integer')
                return
            if command == 'reset':
                if value == 1:
                    # print('Initiating microcontroller reset...')
                    logger.info('Initiating microcontroller reset. Please wait...')
                    string = str(command) + "," + str(value)
                    write_serial_data(serial_connection, string)
                    get_validation(serial_connection, logger)
                elif value == 0:
                    logger.info('Initiating program reset')
                    # print('Initiating program reset...')
                    stop_input = True
                    reset_program = True
                    print('Press ENTER to reset program')
                else:
                    logger.error('Invalid reset failed')
            elif command == 'set-feed-rotations':
                if 0 < value < 101:
                    # print('Changing feed rotation value...')
                    logger.info('Changing feed rotations to ' + str(value) + '. Please wait...')
                    string = str(command) + "," + str(value)
                    write_serial_data(serial_connection, string)
                    get_validation(serial_connection, logger)
                else:
                    logger.error('Bad ' + str(command) + ' input')  #30
                    # print('ERROR: The value for', command, 'must be within range of 1 to 100!')
            elif command == 'set-feed-ppm':
                if 499 < value < 5001:
                    # print('Changing feed CO22 ppm value...')
                    logger.info('Changing feed CO2 ppm threshold ' + str(value) + '. Please wait...')
                    string = str(command) + "," + str(value)
                    write_serial_data(serial_connection, string)
                    get_validation(serial_connection, logger)
                else:
                    logger.error('Bad ' + str(command) + ' input')
                    # print('ERROR: The value for', command, 'must be within range of 500 to 5000!')
            else:
                logger.error('Unrecognized command: ' + command_raw)
                # print('ERROR:', command_raw, 'is not a valid command option!')


def get_serial_ports():
    com_list = serial.tools.list_ports.comports()
    connected = []
    for element in com_list:
        connected.append(element.device)
    return connected


def get_serial_connection(logger, port=None):
    global reset_program, exit_program, stop_input
    if port is None:
        while not exit_program:
            connected_ports = get_serial_ports()
            if connected_ports is not None:
                logger.info('Requesting port input for serial connection')
                # print('Please input a port from the following:')
                logger.info('Connected ports: ' + str(connected_ports))
                # print(get_serial_ports())
                print('Type `reset` to reset the program or type `exit` to exit the program')
                print('Type the port name exactly as it appears below, ex. COM4, to connect to the port')
                for connected_port in connected_ports:
                    print(connected_port)
                print('Input a port from the list of active ports above')
                port = input()
                if port == 'reset':
                    logger.info('Resetting program')
                    stop_input = True
                    reset_program = True
                    break
                elif port == 'exit':
                    logger.info('Exiting program')
                    stop_input = True
                    exit_program = True
                    break
                logger.info('Port input: ' + port)
                try:
                    serial_connection = serial.Serial(port, 115200, timeout=10)
                except serial.SerialException:
                    logger.error('Connection failed with port ' + port)
                    break
                if serial_connection:
                    # print('Establishing serial connection...')
                    logger.info('Serial connection established')
                    return serial_connection
                else:
                    logger.error('Serial connection failed')
                    # print('ERROR: Serial connection failed!')
            else:
                # print('ERROR: No serial port connections detected!')
                logger.error('No serial port connections detected!')
                logger.error('Requesting user to connect microcontroller')
                input('Please connect the microcontroller and press ENTER...')
    else:
        try:
            serial_connection = serial.Serial(port, 115200, timeout=10)
            logger.info('Establishing serial connection with previous port: ' + port)
            # print('Establishing serial connection...')
            return serial_connection
        except serial.SerialException:
            logger.error('Connection failed with port ' + port)


# refresh interval is measured in seconds
def graph_data(logger):
    global historic_date_times, historic_co2_ppm, historic_temps, historic_feed_rotations, data_updated
    global fig, ax1, ax2, ax3
    if data_updated:
        plt.gcf()
        logger.info('Updating graph')
        # matplot lib used for drawing live graphs
        # Plot data since reading began
        # draw CO2 axes
        color = 'tab:red'
        ax1.clear()
        ax1.set_ylabel('CO2 (ppm)', color=color)
        ax1.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'))
        # ax1.axes.get_xaxis().set_visible(False)
        ax1.plot_date(dates.date2num(historic_date_times), historic_co2_ppm, 'b-', color=color)
        ax1.get_yaxis().set_major_locator(ticker.LinearLocator(numticks=4))
        ax1.get_xaxis().set_major_locator(ticker.LinearLocator(numticks=8))
        ax1.grid(b=True, which='major', linestyle='-')

        # draw temp axes
        ax2.clear()
        color = 'tab:blue'
        ax2.set_ylabel('temp (C)', color=color)  # we already handled the x-label with ax1
        ax2.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'))
        # ax2.axes.get_xaxis().set_visible(False)
        ax2.plot_date(dates.date2num(historic_date_times), historic_temps, 'b-', color=color)
        ax2.get_yaxis().set_major_locator(ticker.LinearLocator(numticks=4))
        ax2.get_xaxis().set_major_locator(ticker.LinearLocator(numticks=8))
        ax2.grid(b=True, which='major', linestyle='-')

        ax3.clear()
        color = 'tab:green'
        ax3.set_xlabel('datetime')
        ax3.set_ylabel('feed (rotations)', color=color)
        ax3.xaxis.set_major_formatter(dates.DateFormatter('%m-%d %H:%M'))
        # ax3.axes.get_xaxis().set_visible(False)
        ax3.plot_date(dates.date2num(historic_date_times), historic_feed_rotations, 'o', color=color)
        ax3.get_yaxis().set_major_locator(ticker.LinearLocator(numticks=4))
        ax3.get_xaxis().set_major_locator(ticker.LinearLocator(numticks=8))
        ax3.set_ylim(bottom=1)
        ax3.grid(b=True, which='major', linestyle='-')
        # ax3.set_clip_on(False)

        fig.tight_layout()
        fig.autofmt_xdate()
        fig.canvas.draw()
        plt.pause(.1)
        data_updated = False


def main(logger):
    global reset_program, exit_program, stop_input
    # thread is started that runs the main program in the "background"
    logger.info('Program initializing')
    # print('Bioreactor interface initializing...')
    user_setup()
    while not exit_program:
        serial_connection = get_serial_connection(logger)
        if serial_connection:
            print('Please wait for handshake...')
            stop_input = False
            input_thread = threading.Thread(target=queue_keyboard_input)
            input_thread.daemon = True
            input_thread.start()
            while not reset_program and serial_connection:
                handle_data(serial_connection, logger)
                command_line_interface(serial_connection, logger)
                graph_data(logger)
                if exit_program:
                    break
            input_thread.join()
            serial_connection.close()
        if not exit_program:
            print('Resetting...')
        reset_program = False
    plt.close('all')
    logger.info('Exiting program...')
    exit()


if __name__ == '__main__':
    log = start_logger()
    main(log)
