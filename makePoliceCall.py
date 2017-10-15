from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

# Find these values at https://twilio.com/user/account
account_sid = "ACXXXXXXXXXXXXXXXXX"
auth_token = "YYYYYYYYYYYYYYYYYY"

client = Client(account_sid, auth_token)
def initSer():
    app.run(debug=True)

@app.route("/makeText", methods=['GET', 'POST'])
def makeText():
    client.api.account.messages.create(
                to="+1911",#twilio generated number
                from_="+1123456789",#user entered number
                body="There is an intruder at adress, please send people there now.")
    print(message.sid)
