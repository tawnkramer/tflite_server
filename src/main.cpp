#include <zmq.hpp>
#include <string>
#include <iostream>
#include <stdio.h>

int main(int argc, char** argv)
{

//  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    socket.bind ("tcp://*:5555");
    printf("listening for requests on port 5555\n");

    while (true) {
        zmq::message_t request;

        //  Wait for next request from client
        socket.recv (&request);
	char buf[256];
	strncpy(buf, (const char*)request.data(), request.size());
	buf[request.size()] = 0;
        std::cout << "Received: " << buf << std::endl;

        //  Do some 'work'
	printf("doing work\n");

        //  Send reply back to client
        zmq::message_t reply (5);
        memcpy (reply.data (), "World", 5);
        socket.send (reply);
    }

	return 0;
}
