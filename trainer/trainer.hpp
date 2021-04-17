#ifndef TRAINER_HPP
#define TRAINER_HPP

#include "mnist_reader.hpp"
#include "network.hpp"


class Trainer
{
public:

    bool m_create();

    void m_run();

private:

    mnist::Network m_network;
    mnist::MnistReader m_reader;
};


#endif // TRAINER_HPP