from __future__ import print_function
import mimetypes
import pickle
import os.path
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from random import randint
import json


def get_credentials():
    SCOPES = ['https://www.googleapis.com/auth/gmail.send',
              'https://www.googleapis.com/auth/gmail.modify']
    # Specify permissions to send and read/write messages
    # Find more information at:
    # https://developers.google.com/gmail/api/auth/scopes

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    home_dir = os.path.expanduser('~')
    credentials_dir = os.path.join(home_dir, '.bioreactor')
    if not os.path.exists(credentials_dir):
        os.mkdir(credentials_dir)
    credentials_dir = os.path.join(credentials_dir, '.credentials')
    if not os.path.exists(credentials_dir):
        os.mkdir(credentials_dir)
    credentials_dir = os.path.join(credentials_dir, 'token.pickle')
    if os.path.exists(credentials_dir):
        with open(credentials_dir, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(credentials_dir, 'wb') as token:
            pickle.dump(creds, token)

    service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
    return service


# Create a message
def create_message(sender, to, subject, message_text):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def create_message_with_attachment(sender, to, subject, message_text, file):
    message = MIMEMultipart()
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject

    msg = MIMEText(message_text)
    message.attach(msg)

    content_type, encoding = mimetypes.guess_type(file)

    if content_type is None or encoding is not None:
        content_type = 'application/octet-stream'
    main_type, sub_type = content_type.split('/', 1)
    if main_type == 'text':
        fp = open(file, 'rb')
        msg = MIMEText(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'image':
        fp = open(file, 'rb')
        msg = MIMEImage(fp.read(), _subtype=sub_type)
        fp.close()
    elif main_type == 'audio':
        fp = open(file, 'rb')
        msg = MIMEAudio(fp.read(), _subtype=sub_type)
        fp.close()
    else:
        fp = open(file, 'rb')
        msg = MIMEBase(main_type, sub_type)
        msg.set_payload(fp.read())
        fp.close()
    filename = os.path.basename(file)
    msg.add_header('Content-Disposition', 'attachment', filename=filename)
    message.attach(msg)

    return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}


def send_message(service, body):
    message = None
    while message is None:
        message = (service.users().messages().send(userId="me", body=body).execute())
    message_id = message['id']
    return message_id


def get_contacts(service):
    while True:
        print('Please enter the name of the notifications contact')
        name = input('>')
        while True:
            print('Please enter primary email address')
            email = input('>')
            otp = randint(100000, 999999)
            message = create_message('RasPi@Bioreactor', email, 'Authentication',
                                     'Hello {0}\nPlease use this code to authenticate bioreactor notifications '
                                     'contact address: {1}'.format(
                                         name, str(otp)))
            message_id = send_message(service, message)
            print('A 6-digit OTP has been emailed to your provided email address with message ID: {0}\nPlease check '
                  'your email messages and enter your 6 digit OTP to confirm'.format(str(message_id)))
            user_otp = input('>')
            if otp == int(user_otp):
                print('Authenticated')
                break
            else:
                print('Authentication failed\nPlease enter a working email address')
        while True:
            print('Please enter an email to text address')
            sms_email = input('>')
            otp = randint(100000, 999999)
            message = create_message('RasPi@Bioreactor', sms_email, 'Authentication',
                                     'Hello {0},\nPlease use this code to authenticate bioreactor notifications '
                                     'contact address: {1}'.format(
                                         name, str(otp)))
            message_id = send_message(service, message)
            print('A 6-digit OTP has been emailed to your provided sms address with message ID: {0}\nPlease check '
                  'your text messages and enter your 6 digit OTP to confirm'.format(str(message_id)))
            user_otp = input('>')
            if otp == int(user_otp):
                print('Authenticated')
                break
            else:
                print('Authentication failed\nPlease enter a working email address')
        print('Please confirm the following contact information')
        print('Name: {0}\nEmail: {1}\nSMS Email: {2}'.format(name, email, sms_email))
        answer = input('Confirm? (Y,n) >')
        if answer.lower() == 'y':
            print('Confirmed')
            break
        else:
            print('User input rejected')
    user_contact_dict = {
        'user': name,
        'email': email,
        'sms_email': sms_email
    }
    print('Attempting to add the following user')
    print(user_contact_dict)
    return user_contact_dict


def write_user_json(user_dict):
    home_dir = os.path.expanduser('~')
    config_dir = os.path.join(home_dir, '.bioreactor')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    config_dir = os.path.join(config_dir, '.config')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    config_dir = os.path.join(config_dir, 'users.json')
    if not os.path.exists(config_dir):
        with open(config_dir, 'w') as write_file:
            new_dict = dict()
            new_dict[0] = user_dict
            json.dump(new_dict, write_file, indent=1)
    else:
        with open(config_dir) as write_file:
            data = json.load(write_file)
        for key in data:
            for user_value in user_dict:
                for data_value in data[key]:
                    if user_dict[user_value] == data[key][data_value]:
                        print('Write failed\nThis user has matching values in an existing record')
                        return False
        data[len(data.values())] = user_dict
        with open(config_dir, 'w') as write_file:
            json.dump(data, write_file, indent=1)


def get_users_json():
    home_dir = os.path.expanduser('~')
    config_dir = os.path.join(home_dir, '.bioreactor')
    config_dir = os.path.join(config_dir, '.config')
    config_dir = os.path.join(config_dir, 'users.json')
    try:
        with open(config_dir) as write_file:
            data = json.load(write_file)
            return data
    except FileNotFoundError:
        return False


def get_user_email_list():
    data = get_users_json()
    emails = []
    for key in data:
        emails.append(data[key].get('email'))
    return emails


def get_user_sms_list():
    data = get_users_json()
    sms_emails = []
    for key in data:
        sms_emails.append(data[key].get('sms_email'))
    return sms_emails


def send_mass_email_with_log(service, message):
    email_list = get_user_email_list()
    home_dir = os.path.expanduser('~')
    log_file = os.path.join(home_dir, '.bioreactor')
    log_file = os.path.join(log_file, '.log')
    log_file = os.path.join(log_file, 'log.log')
    for address in email_list:
        email = create_message_with_attachment('RasPi@Bioreactor', address, 'Bioreactor Log', message, log_file)
        message_id = send_message(service, email)
        print('Log file emailed to {0}, message ID: {1}'.format(address, message_id))
    return True


def send_mass_sms(service, subject, message):
    sms_list = get_user_sms_list()
    for address in sms_list:
        sms = create_message('Raspi@Bioreactor', address, subject, message)
        message_id = send_message(service, sms)
        print('{0} message texted to {1}, message ID: {2}'.format(subject, address, message_id))
    return True


def user_setup():
    service = get_credentials()
    if not get_users_json():
        user = get_contacts(service)
        if write_user_json(user):
            print('User added to JSON file')
    while True:
        print('Would you like to add a user contact entry? (Y/n)')
        answer = input('>')
        if answer.lower() == 'n':
            break
        if answer.lower() == 'y':
            user = get_contacts(service)
            if write_user_json(user):
                print('User added to JSON file')
        else:
            print('Input failed\nPlease enter `y` for yes or `n` for no')
