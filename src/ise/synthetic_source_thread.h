/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_source_thread_h_
#define _synthetic_source_thread_h_

#include <QThread>

class Synthetic_source;
class Frame;

class Synthetic_source_thread : public QThread {
public:
    Q_OBJECT
    ;
    
public:
    Synthetic_source_thread (int width, int height);
    virtual ~Synthetic_source_thread ();
    void set_synthetic_source (Synthetic_source *ss);
    unsigned int width;
    unsigned int height;

protected:
    virtual void run();

public:
    Synthetic_source *ss;

signals:
    void frame_ready (Frame* f, int width, int height);
    
};

#endif
