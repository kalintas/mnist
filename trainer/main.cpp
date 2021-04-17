#include "trainer.hpp"


int main() 
{
    Trainer trainer;

    if (!trainer.m_create()) return -1;

    trainer.m_run();

}