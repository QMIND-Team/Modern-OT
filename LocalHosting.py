#Import Packages for creating a simple http local host
import http.server
import socketserver

#Specify the server port of the local host
PORT = 8888

#Creates http handler snd server
Handler = http.server.SimpleHTTPRequestHandler

httpd = socketserver.TCPServer(("", PORT), Handler)

print("serving at port", PORT)
httpd.serve_forever()