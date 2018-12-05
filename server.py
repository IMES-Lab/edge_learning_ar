
import socket
import pickle
from _thread import *
import threading

#print_lock = threading.Lock()


# thread
def threaded(c):
    while True:

        # data received from client 1024 byte
        data = pickle.loads(c.recv(1024))
        if not data:
            print('Connection terminated')

            # lock released on exit
            #print_lock.release()
            break

        # reverse the given string from client
        data = data[::-1]

        # send back reversed string to client
        c.send(pickle.dumps(data))

    # connection closed
    c.close()


def Main():
    host = ""
    port = 7070
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to post", port)

    # put the socket into listening mode
    s.listen(5)
    print("server Initiated")

    # a forever loop until client wants to exit
    while True:
        # establish connection with client
        c, addr = s.accept()

        # lock acquired by client
        #print_lock.acquire()
        print('Connection from :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()


if __name__ == '__main__':
    Main()
