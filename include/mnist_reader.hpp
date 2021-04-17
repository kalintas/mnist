#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include <string_view>
#include <iostream>
#include <vector>
#include <type_traits>

#include "utils.hpp"

namespace mnist
{

    class MnistReader
    {
    public:

        ~MnistReader()
        {
            if (m_imageFile) std::fclose(m_imageFile);
            if (m_labelFile) std::fclose(m_labelFile);
        }

        bool m_create(const std::string_view imageFilePath, const std::string_view labelFilePath)
        {
            m_imageFile = std::fopen(imageFilePath.data(), "rb");
            m_labelFile = std::fopen(labelFilePath.data(), "rb");

            if (!m_imageFile || !m_labelFile)
            {
                std::cerr << "Cannot load training files\n";
                return false;
            }

            ReadBigEndianUInt(m_imageFile); // skip the file header
            m_imageCount  = ReadBigEndianUInt(m_imageFile);
            m_imageWidth  = ReadBigEndianUInt(m_imageFile);
            m_imageHeight = ReadBigEndianUInt(m_imageFile);

            // skip 8 byte junk
            ReadBigEndianUInt(m_labelFile);
            ReadBigEndianUInt(m_labelFile);

            return true;
        }

        template<typename ContainerT = std::vector<double>>
        void m_getNextImage(ContainerT& image, std::uint8_t& label)
        {   
            using Type = typename ContainerT::value_type;

            image.resize(m_imageWidth * m_imageHeight);
    
            for (auto& color : image)
            {

                if constexpr (std::is_integral_v<Type>)
                {
                    color = ReadNext<std::uint8_t>(m_imageFile);
                }
                else
                {
                    color = static_cast<Type>(ReadNext<std::uint8_t>(m_imageFile)) / 255.0; 
                }
            }

            label = ReadNext<std::uint8_t>(m_labelFile);

            ++m_currentImage;
        }

        [[nodiscard]] constexpr operator bool() const noexcept { return m_currentImage < m_imageCount; }
        
        [[nodiscard]] constexpr auto m_imageSize() const noexcept { return m_imageWidth * m_imageHeight; }

    private:

        std::FILE* m_imageFile = nullptr;
        std::FILE* m_labelFile = nullptr;

        std::size_t m_imageCount   = 0;
        std::size_t m_imageWidth   = 0;
        std::size_t m_imageHeight  = 0;
        std::size_t m_currentImage = 0;
    };

}


#endif // MNIST_READER_HPP