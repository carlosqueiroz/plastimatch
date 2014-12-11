#include <stdio.h>


/* Simple test */
typedef void (*B) (int);

void c (int x) {
    printf ("x = %d\n", x);
}

template <B b> void a (int x) {
    b(x);
}

/* More complex */
// This is illegal, typedef's can't be templated
// typedef template void<F> (*E) (F);
//void g<int> (int x) {
//    printf ("x = %d\n", x);
//}

template <typename F>
class G
{
public:
    static void g (F x) {
        printf ("x = %d\n", x);
    }
};


template < class G > void d (int x) {
    G::g (x);
}

/* Even more complex */
template < class J, template<class J> class I> void h (J x) {
    I<J>::g (x);
}

int main 
(
    int argc,
    char* argv[]
)
{
    int x = 1;
    a<c>(x);
    G<int>::g (x);
    d< G<int> >(x);
    h< int, G >(x);
}
