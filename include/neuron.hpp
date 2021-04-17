#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <cstring>
#include <cmath>

#include "utils.hpp"

namespace mnist
{
    class Neuron;

    using Layer = std::vector<Neuron>;

    class Neuron
    {
    public:

        Neuron(const std::size_t nextLayerCount, const std::size_t neuronIndex)
            : m_neuronIndex(neuronIndex)
        {
            m_connections.resize(nextLayerCount);

            for (auto& [weight, deltaWeight] : m_connections)
            {
                weight = s_getRandomWeight();
                deltaWeight = 0;
            }
        }

        Neuron() noexcept = default;

        void m_feedForward(const Layer& prevLayer)
        {
            double sum = 0.0;

            for (const auto& neuron : prevLayer)
            {
                sum += neuron.m_neuronValue * neuron.m_connections[m_neuronIndex].m_weight;
            }

            m_neuronValue = s_activate(sum / static_cast<double>(prevLayer.size()));
        }

        constexpr void m_calculateOutputGradients(const double target) noexcept
        {
            const auto diff = target - m_neuronValue;

            m_gradient = diff * s_activateDerivative(m_neuronValue);
        }

        void m_calculateHiddenGradients(const Layer& nextLayer) noexcept
        {
            double sum = 0.0;

            for (std::size_t i = 0; i < nextLayer.size() - 1; ++i)
            {
                sum += m_connections[i].m_weight * nextLayer[i].m_gradient;
            }

            m_gradient = sum * s_activateDerivative(m_neuronValue);
        }

        void m_updateInputWeights(Layer& prevLayer) noexcept
        {
            constexpr auto eta   = 0.3;
            constexpr auto alpha = 0.45;
            
            for (auto& neuron : prevLayer)
            {
                const auto oldDeltaWeight = neuron.m_connections[m_neuronIndex].m_deltaWeight;

                const auto newDeltaWeight = eta * neuron.m_neuronValue * m_gradient
                    + alpha * oldDeltaWeight;

                neuron.m_connections[m_neuronIndex].m_deltaWeight = newDeltaWeight;
                neuron.m_connections[m_neuronIndex].m_weight     += newDeltaWeight;
            }
        }

        constexpr Neuron& operator= (const double value) noexcept
        {
            m_neuronValue = value;
            return *this;
        }

        [[nodiscard]] constexpr operator double() const noexcept
        {
            return m_neuronValue;
        }

    public:

        void m_writeToFile(std::FILE* file) const
        {
            WriteNext(file, m_neuronValue);
            WriteNext(file, m_neuronIndex);
            WriteNext(file, m_gradient   );

            WriteContainer(file, m_connections);
        }

        void m_readFromFile(std::FILE* file)
        {
            ReadNext(file, m_neuronValue);
            ReadNext(file, m_neuronIndex);
            ReadNext(file, m_gradient   );

            ReadContainer(file, m_connections);
        }

    public:

        double m_neuronValue = 0.0;

    private:


        std::size_t m_neuronIndex = 0;

        struct Connection
        {
            double m_weight;
            double m_deltaWeight;
        };

        std::vector<Connection> m_connections;

        double m_gradient = 0.0;

    private:

        [[nodiscard]] static double s_getRandomWeight() noexcept
        {
            return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
        }

        [[nodiscard]] static double s_activate(const double x)
        {
            return std::tanh(x);
        }

        [[nodiscard]] static constexpr double s_activateDerivative(const double x) noexcept
        {
            return 1.0 - x * x;
        }

    };

}


#endif // NEURON_HPP