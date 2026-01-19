#include "core/std.hpp"
#include <iostream>
#include"core/core.hpp"

void print()
{
	std::cout << "Hello sub-library 1!" << std::endl;
}

sm::Core* m_Core;
namespace sm {
    Core::Core()
    {
        
    }
    Core* Core::get_init()
    {
        if (m_Core == NULL)
        {
            m_Core = new Core();
        }
        return m_Core;
    }
}