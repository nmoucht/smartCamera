#!/usr/bin/env python
from vision import isPerson, isFace
from twilio.rest import Client
from messageReceiver import initServ, sms
from facialRec import initial
# Find these values at https://twilio.com/user/account
account_sid = "ACXXXXXXXXXXXXXXXXX"
auth_token = "YYYYYYYYYYYYYYYYYY"

client = Client(account_sid, auth_token)

def main():
	cap = cv2.VideoCapture(0)
	while(True):
    	ret, frame = cap.read()
   		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   		if(isPerson(frame)):
   			arr,areFaces=isFace(frame)
   			if(areFaces):
   				param=False
   				nameOfPerson=""
   				image=frame
   				for imgs in arr:
   					name,isKown=initial(imgs)
   					if(isKown):
   						param=True
   						nameOfPerson=name
   						image=imgs
   				if(param):
   					#send text that blank came home
   					client.api.account.messages.create(
    				to="+12316851234",#twilio generated number
    				from_="+19082555551",#user entered number
    				body="Hello there!"+nameOfPerson+" just came home.")
    				print(message.sid)
   				else:
   					client.api.account.messages.create(
    				to="+12316851234",#twilio generated number
    				from_="+19082555551",#user entered number
    				body="There is an unknown person in the house. " 
    				"You have 1 minute to cancel the police call. "
    				"If you want to cancel, text back Cancel. "
    				"If you want to call, text back Call. "
    				"Here is a picture of what caused this message:"
    				media=image)
    				print(message.sid)
    				initServ()
   					#send text that unknown person 
   					#entered house send picture and wait one minute then call cops

	cap.release()

if __name__ == "__main__":
	main()