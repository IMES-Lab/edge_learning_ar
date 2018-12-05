import socket
import pickle

def Main():
	host = '127.0.0.1'
	port = 7070

	s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

	# connect to server on local computer
	s.connect((host,port))

	# message you send to server
	message = "TEST MESSAGE"
	while True:

		# message sent to server
		s.send(pickle.dumps(message))

		# messaga received from server
		data = s.recv(1024)

		# print the received message
		# here it would be a reverse of sent message
		print('Received from the server :',str(pickle.loads(data)))

		# ask the client whether he wants to continue
		ans = input('\nDo you want to continue(y/n) :')
		if ans == 'y':
			continue
		else:
			break
	# close the connection
	s.close()

if __name__ == '__main__':
	Main()
