#include "trainer.hpp"

#include <array>
#include <string>


bool Trainer::m_create()
{   
    if (!m_reader.m_create("../res/train-images.idx3-ubyte", "../res/train-labels.idx1-ubyte"))
    {
        std::cerr << "Cannot Find train samples\n";
        return false;
    }

    if (!m_network.m_readFromFile("../res/train-data"))
    {
        // layer layout for neural network
        std::array<std::size_t, 3> layout{ m_reader.m_imageSize(), 100, 10 };

        m_network.m_create(layout);
    }
    
    return true;
}

void Trainer::m_run()
{
    std::vector<double> image;
    std::vector<double> results;

    std::size_t pass = 0;
    std::size_t success = 0;

    while (m_reader)
    {   
        std::uint8_t targetValue;

        m_reader.m_getNextImage(image, targetValue);

        m_network.m_feedForward(image);

        m_network.m_getResults(results);

        auto result = std::max_element(results.begin(), results.end()) - results.begin();

        if (result == targetValue) ++success;
        
        std::array<double, 10> targetValues{};

        targetValues[targetValue] = 1.0;

        m_network.m_doBackpropagation(targetValues);
    
        if (++pass % 15 == 0) // dont spam the console
        {   
            std::cout << "pass = " << pass << "\n";
            std::cout << "target value = " << static_cast<int>(targetValue) << ", result value = " << result << "\n";
            std::cout << "overall success rate = " << static_cast<double>(success) / static_cast<double>(pass) << "\n";
        }
    }
    
    std::cout << "Done\n";

    m_network.m_writeToFile("../res/train-data");
}

