#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdio>
#include <bit>

namespace mnist
{
    [[nodiscard]] constexpr std::uint32_t SwapEndiannes(const std::uint32_t val) noexcept
    {
        return (val >> 24) | ((val >> 8) & 0xFF00) | ((val << 8) & 0xFF0000) | (val << 24);
    }

    template<typename Type>
    Type ReadNext(std::FILE* file)
    {
        Type buffer;

        std::fread(&buffer, sizeof(Type), 1, file);

        return buffer;
    }

    template<typename Type>
    void ReadNext(std::FILE* file, Type& buffer)
    {
        std::fread(&buffer, sizeof(Type), 1, file);
    }
    
    template<typename Type>
    void WriteNext(std::FILE* file, const Type& value)
    {
        std::fwrite(&value, sizeof(Type), 1, file);
    }

    inline std::uint32_t ReadBigEndianUInt(std::FILE* file)
    {
        auto buffer = ReadNext<std::uint32_t>(file);

        if constexpr (std::endian::native != std::endian::big)
        {
            // swap big endian 4 bytes to little endian
            return SwapEndiannes(buffer);
        }
        else
        {   
            return buffer;
        }
    }


    template<typename ContainerT>
    void WriteContainer(std::FILE* file, const ContainerT& buffer)
    {
        WriteNext(file, buffer.size());

        std::fwrite(buffer.data(), sizeof(typename ContainerT::value_type), buffer.size(), file);
    }

    template<typename ContainerT>
    void ReadContainer(std::FILE* file, ContainerT& buffer)
    {   
        buffer.resize(ReadNext<typename ContainerT::size_type>(file));

        std::fread(buffer.data(), sizeof(typename ContainerT::value_type), buffer.size(), file);
    }

}



#endif // UTILS_HPP