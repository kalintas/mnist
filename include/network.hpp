#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <algorithm>
#include <string_view>

#include "neuron.hpp"

namespace mnist
{

    class Network
    {
    public:

        template<typename ContainerT>
        void m_create(const ContainerT& layerLayout)
        {
            // create layers according to layerLayout
            m_layers.clear();
            m_layers.reserve(layerLayout.size());

            for (auto it = layerLayout.begin(); it != layerLayout.end(); ++it)
            {
                const auto layerCount = *it + 1; // + 1 for bias
                const auto nextLayerCount = (it + 1 == layerLayout.end()) ? 0 : *(it + 1);
                
                Layer layer;
                layer.reserve(layerCount);

                for (std::size_t i = 0; i < layerCount; ++i)
                {
                    layer.emplace_back(nextLayerCount, i);
                }

                layer.back() = 1.0; // set bias neuron value to 1

                m_layers.emplace_back(std::move(layer));
            }
        }

        template<typename ContainerT>
        void m_feedForward(const ContainerT& inputValues)
        {
            auto& inputLayer = m_layers.front();

            if (inputValues.size() != inputLayer.size() - 1)
            {
                std::cerr << "Network::m_feedForward wrong parameters\n";
                return;
            }

            // copy input values to input layer neuron values
            std::copy(inputValues.begin(), inputValues.end(), inputLayer.begin());

            // iterate trough layers starting after the input layer
            for (auto it = m_layers.begin() + 1; it != m_layers.end(); ++it)
            {
                std::for_each(it->begin(), it->end() - 1, // dont include bias neuron (it has no input) 
                    [it] (auto& neuron) 
                    {  
                        // pass the previous layer
                        neuron.m_feedForward(*(it - 1));
                    });
            }
        }

        template<typename ContainerT>
        void m_doBackpropagation(const ContainerT& targetValues)
        {   
            auto& outputLayer = m_layers.back();

            if (targetValues.size() != outputLayer.size() - 1)
            {
                std::cerr << "Network::m_doBackpropagation wrong parameters\n";
                return;
            }

            auto targetIt = targetValues.begin();

            // calculate gradients for output layer
            for (auto it = outputLayer.begin(); it != outputLayer.end() - 1; ++it, ++targetIt)
            {
                it->m_calculateOutputGradients(*targetIt);
            }

            // calculate gradients for hidden layers
            for (auto rIt = m_layers.rbegin() + 1; rIt != m_layers.rend() - 1; ++rIt)
            {
                const auto& nextLayer = *(rIt - 1);

                std::for_each(rIt->begin(), rIt->end(), 
                    [&] (auto& neuron) { neuron.m_calculateHiddenGradients(nextLayer); });
            }

            // update weights according to gradient values
            for (auto rIt = m_layers.rbegin(); rIt != m_layers.rend() - 1; ++rIt)
            {
                auto& prevLayer = *(rIt + 1);

                std::for_each(rIt->begin(), rIt->end() - 1, 
                    [&] (auto& neuron) { neuron.m_updateInputWeights(prevLayer); });
            }
        }

        template<typename ContainerT>
        void m_getResults(ContainerT& result) const
        {       
            const auto& outputLayer = m_layers.back();
            
            result.resize(outputLayer.size() - 1);

            std::copy(outputLayer.begin(), outputLayer.end() - 1, result.begin());
        }

    public:
        
        bool m_writeToFile(const std::string_view filePath) const
        {
            auto file = std::fopen(filePath.data(), "wb");

            if (!file) return false;

            WriteNext(file, m_layers.size());

            for (const auto& layer : m_layers)
            {   
                WriteNext(file, layer.size());
            }

            for (const auto& layer : m_layers)
            {   
                for (const auto& neuron : layer)
                {
                    neuron.m_writeToFile(file);
                }
            }

            std::fclose(file);

            return true;
        }

        bool m_readFromFile(const std::string_view filePath)
        {   
            auto file = std::fopen(filePath.data(), "rb");

            if (!file) return false;

            m_layers.resize(ReadNext<std::size_t>(file));

            for (auto& layer : m_layers)
            {
                layer.resize(ReadNext<std::size_t>(file));
            }

            for (auto& layer : m_layers)
            {
                for (auto& neuron : layer)
                {
                    neuron.m_readFromFile(file);
                }
            }

            std::fclose(file);

            return true;
        }


    private:

        std::vector<Layer> m_layers;
    };

}

#endif // NETWORK_HPP