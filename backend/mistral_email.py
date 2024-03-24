import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_email_with_content(email_content, phone_receiver_address):
    # Your email credentials
    sender_address = 'telephonique.mistral@gmail.com'
    sender_pass = 'vhwmkceggicykpnf'
    receiver_address = phone_receiver_address

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Notes from Telephonique'  # The subject line

    # The body of the mail with the content passed from the command line
    mail_content = f'''Hey, you requested a note from Telephonique. Here it is!

-----

{email_content}
'''

    message.attach(MIMEText(mail_content, 'plain'))

    # Create SMTP session for sending the mail
    try:
        session = smtplib.SMTP('smtp.gmail.com', 587)  # use Gmail with port
        session.starttls()  # enable security
        session.login(sender_address, sender_pass)  # login with mail_id and password
        text = message.as_string()
        session.sendmail(sender_address, receiver_address, text)
        session.quit()
        print('Mail Sent')
    except Exception as e:
        print(f'Failed to send email: {e}')
