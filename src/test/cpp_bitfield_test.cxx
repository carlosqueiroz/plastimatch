#include <stdio.h>


enum bitmask : unsigned char {
    C0 = 1 << 0,
    C1 = 1 << 1,
    C2 = 1 << 2
};
constexpr bitmask operator|(bitmask X, bitmask Y) {
    return static_cast<bitmask>(
        static_cast<unsigned char>(X) | static_cast<unsigned char>(Y));
}


int main (int argc, char* argv[])
{
    bitmask c = C0;
    c = C1 | C2;
    if (c == C2) {
        printf ("Oops.\n");
    }
    if (c & C2) {
        printf ("Yep.\n");
    }
    return 0;
}
