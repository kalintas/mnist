
#pragma warning(push, 0)
#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#pragma warning(pop)

#include "network.hpp"


class Renderer : public olc::PixelGameEngine
{
public:

    bool m_construct()
    {
        return Construct(32 * 4, 32 * 4, 6, 6, false, true);
    }

    [[nodiscard]] olc::vi2d m_getPos(const std::int32_t i)
    {
        return { (2 + (i % m_width)) * 4, (2 + (i / m_width)) * 4 };
    }
    
    bool OnUserCreate() override
    {
        if (!m_network.m_readFromFile("../res/train-data"))
        {
            std::cerr << "Cannot find train data\n";
            return false;
        }

        m_image.resize(m_width * m_height);

        sAppName = "mnist";

        m_updateScreen();

        return true;
    }

    bool OnUserUpdate(float) override
    {
        m_handleMouse(0);
        m_handleMouse(1);
        
        if (GetKey(olc::Key::SPACE).bPressed)
        {
            std::fill(m_image.begin(), m_image.end(), 0.0);
            m_updateScreen();
        }


        return !GetKey(olc::Key::ESCAPE).bPressed;
    }

private:

    mnist::Network m_network;

    std::vector<double> m_image;    
    std::uint8_t m_currentLabel = 0;

    std::int32_t m_width  = 28;
    std::int32_t m_height = 28;

private:

    void m_handleMouse(const std::uint32_t button)
    {
        if (GetMouse(button).bPressed || GetMouse(button).bHeld)
        {
            const auto mousePos = GetMousePos() / 4 - olc::vi2d{ 2, 2 };

            if (m_setPixel(mousePos, 0.7, button))
            {
                m_setPixel(mousePos + olc::vi2d{  1,  0 }, 0.05, button);
                m_setPixel(mousePos + olc::vi2d{ -1,  0 }, 0.05, button);
                m_setPixel(mousePos + olc::vi2d{  0,  1 }, 0.05, button);
                m_setPixel(mousePos + olc::vi2d{  0, -1 }, 0.05, button);

                m_updateLabel();
                m_updateScreen();
            }
        }
    }

    bool m_setPixel(const olc::vi2d vec, const double amount, const bool decrease)
    {        
        if (vec.x >= 0 && vec.y >= 0 && vec.x < m_width && vec.y < m_height)
        {
            auto& color = m_image.at(vec.y * m_width + vec.x);

            if (decrease)
            {
                color = std::max(0.0, color - amount);
            }
            else
            {
                color = std::min(1.0, color + amount);
            }

                        
            return true;
        }

        return false;
    }

    void m_updateLabel()
    {
        m_network.m_feedForward(m_image);

        std::vector<double> results;

        m_network.m_getResults(results);

        m_currentLabel = std::max_element(results.begin(), results.end()) - results.begin();
    }

    void m_updateScreen()
    {
        Clear(olc::RED);

        for (std::int32_t i = 0; i < m_image.size(); ++i)
        {   
            const auto color = static_cast<std::uint8_t>(m_image.at(i) * 255.0);

            FillRect(m_getPos(i), { 4, 4 }, { color, color, color });        
        }

        DrawString(0, ScreenHeight() - 8, "result = " + std::to_string(m_currentLabel));
    }

};


int main()
{
    Renderer renderer;

    if (!renderer.m_construct()) return -1;
    renderer.Start();
}



