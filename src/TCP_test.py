import socket

def listen_for_gamestate(port=12345):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    conn.setblocking(False)
    buffer = ""
    while True:
        try:
            data = conn.recv(4096).decode()
            if not data:
                return
            buffer += data
        except BlockingIOError:
            pass
        
        if "\n" in buffer:
            lines = buffer.split("\n")
            buffer = lines[-1]  # keep incomplete trailing data
            yield lines[-2]     # yield the last complete line

if __name__ == "__main__":

   action = 11
   action = ((a := int(action//9)), action-a*9)
   print(action)