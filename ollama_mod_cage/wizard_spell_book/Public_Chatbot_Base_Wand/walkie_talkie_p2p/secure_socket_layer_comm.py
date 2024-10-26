"""
#TODO Import python peer 2 peer pyp2p, for decentralized
encrypted nodes

created on: 5/23/2024
by @LeoBorcherding
"""

# server.py
import socket
import ssl

class secure_socket_layer_server_example:
    def __init__(self, host='localhost', port=48956):
        self.server_address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.server_address)
        self.context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.context.load_cert_chain(certfile='server.crt', keyfile='server.key')
        self.connection = None

    def start(self):
        self.sock.listen(1)
        while True:
            print('waiting for a connection')
            raw_connection, client_address = self.sock.accept()
            self.connection = self.context.wrap_socket(raw_connection, server_side=True)
            self.handle_connection(client_address)

    def handle_connection(self, client_address):
        try:
            print('connection from', client_address)
            while True:
                data = self.connection.recv(1024)
                print('received {!r}'.format(data))
                if data:
                    print('sending data back to the client')
                    self.connection.sendall(data)
                else:
                    print('no data from', client_address)
                    break
        finally:
            self.connection.close()

    def send_message(self, message):
        if self.connection:
            print('sending {!r}'.format(message))
            self.connection.sendall(message.encode())
        else:
            print('No connection to send message')

if __name__ == "__main__":
    server = secure_socket_layer_server_example()
    server.start()


class class_usage_example:
    def __init__(self, server):
        self.server = server

    def do_something(self):
        # Do something...
        self.server.send_message("Hello from AnotherClass!")

if __name__ == "__main__":
    server = secure_socket_layer_server_example()
    another_class = class_usage_example(server)
    another_class.do_something()
    server.start()