#include <zmq.hpp>
#include <string>
#include <iostream>
#include <stdio.h>
#include "tflite_model.h"
#include <vector>
#include <string>

using namespace std;

void show_usage()
{
    printf("Usage: tflite_serve --model <model.tflite> [--port <int>] [--num_threads <int>]\n");    
}

int main(int argc, char** argv)
{
    string filename;
    int port = 5555;
    int num_threads = 1;

    if(argc == 1)
    {
        show_usage();
        exit(-1);
    }

    for(int i = 0; i < argc; i++)
    {
        char* arg = argv[i];

        if (strcmp(arg, "--model") == 0 && i < argc - 1)
        {
            filename = argv[i + 1];
        }

        if (strcmp(arg, "--port") == 0 && i < argc - 1)
        {
            port = atoi(argv[i + 1]);
        }

        if (strcmp(arg, "--num_threads") == 0 && i < argc - 1)
        {
            num_threads = atoi(argv[i + 1]);
        }

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
        {
            show_usage();
            exit(1);
        }
    }

    //Load TFlite model
    TFLiteModel* pModel = new TFLiteModel();

    if(!pModel->Load(filename.c_str()))
        exit(1);

    printf("Loaded: %s\n", filename.c_str());
    pModel->ShowInputs();
    printf("Wants: %zu bytes\n", pModel->GetInputSize());

    pModel->SetNumThreads(num_threads);
    printf("Setting num threads: %d\n", num_threads);

    //  Prepare our context and socket
    zmq::context_t context (1);
    zmq::socket_t socket (context, ZMQ_REP);
    char connect_str[256];
    sprintf(connect_str, "tcp://*:%d", port);
    socket.bind (connect_str);

    printf("listening for requests on port: %d\n", port);

    char buf[256];

    while (true) {
        zmq::message_t request;

        //  Wait for next request from client
        socket.recv (&request);

        string result;

        if(pModel->Inference(request.data(), request.size(), result))
        {
            pModel->GetResultJson(result);   
        }

        //  Send reply back to client
        zmq::message_t reply (result.size());
        memcpy (reply.data (), result.c_str(), result.size());
        socket.send (reply);
    }

    delete pModel;

    return 0;
}
