""" FileSharingNode.py

        a class for creating a custom file sharing node with inbound and outbound messaging.
    This file sharing node is a custom node build with the macsnoeren/python-p2p-network 
    repository on github, and ultimately utilizes this package to create articulatable
    peer to peer consensus networks for ollama agent roll cage users.
"""
from p2pnetwork.node import Node
import subprocess

class FileSharingNode (Node):

    def __init__(self, host, port, id=None, callback=None, max_connections=0, chatbot_model=None):
        super(FileSharingNode, self).__init__(host, port, id, callback, max_connections)
        self.chatbot_model = chatbot_model
    def outbound_node_connected(self, connected_node):
        print("outbound_node_connected: " + connected_node.id)
        
    def inbound_node_connected(self, connected_node):
        print("inbound_node_connected: " + connected_node.id)

    def inbound_node_disconnected(self, connected_node):
        print("inbound_node_disconnected: " + connected_node.id)

    def outbound_node_disconnected(self, connected_node):
        print("outbound_node_disconnected: " + connected_node.id)

    def node_message(self, connected_node, data):
        print("node_message from " + connected_node.id + ": " + str(data))
        
    def node_disconnect_with_outbound_node(self, connected_node):
        print("node wants to disconnect with oher outbound node: " + connected_node.id)
        
    def node_request_to_stop(self):
        print("node is requested to stop!")

    def start_node(self, host="127.0.0.1", port=9876):
        self.host = host
        self.port = port
        # Start a new cmd process that runs the node
        subprocess.Popen(["start", "cmd", "/k", "python", "path_to_your_script.py", str(host), str(port)], shell=True)

    def node_message(self, connected_node, data):
        print("node_message from " + connected_node.id + ": " + str(data))
        # Pass the received message to your chatbot model
        response = self.chatbot_model.process_prompt(data)
        print("Chatbot response: " + response)