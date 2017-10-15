# /usr/bin/env python
# Download the twilio-python library from twilio.com/docs/libraries/python
from flask import Flask
from twilio.twiml.messaging_response import MessagingResponse
from makePoliceCall import initSer, makeText
app = Flask(__name__)

def initServ():
	app.run(debug=True)
@app.route('/sms', methods=['POST'])
def sms():
    number = request.form['From']
    message_body = request.form['Body']
    resp = twiml.Response()
    if (message_body.lower()==_"cancel"):
    	resp.message("Thanks, the call has been cancelled")
    else:
    	initSer()
    return str(resp)
