/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cbuf_h_
#define _cbuf_h_

#include <list>
#include <stdio.h>

class Frame;

class Cbuf {
public:
    Cbuf ();
    ~Cbuf ();
    int init_queue (unsigned int idx, unsigned int num_frames, 
        unsigned int size_x, unsigned int size_y);
public:
    unsigned long idx;
    unsigned long num_frames;
    unsigned long size_x;
    unsigned long size_y;
    std::list<Frame*> frames;
    unsigned long writable;
    unsigned long waiting_unwritten;
    unsigned long dropped;
    std::list<Frame*> empty;
    std::list<Frame*> waiting;
    Frame* write_ptr;
    Frame* display_ptr;
    Frame* internal_grab_ptr;
#ifdef _WIN32
    CRITICAL_SECTION cs;
#endif
};


#if defined (commentout)
void cbuf_init (IseFramework* ig);
int cbuf_init_queue (IseFramework* ig, unsigned int idx, unsigned int num_frames);
void cbuf_append_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_insert_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_append_empty (CBuf* cbuf, Frame* new_frame);
Frame* cbuf_get_empty_frame (CBuf* cbuf);
Frame* cbuf_get_any_frame (CBuf* cbuf, int flush_dark);
unsigned long cbuf_queuelen (CBuf* cbuf, FrameQueue* queue);
int cbuf_shutdown (IseFramework* ig);
void cbuf_shutdown_queue (IseFramework* ig, CBuf* cbuf);
Frame* cbuf_display_lock_newest_frame (CBuf* cbuf);
Frame* cbuf_display_lock_newest_bright (CBuf* cbuf);
Frame* cbuf_display_lock_oldest_frame (CBuf* cbuf);
Frame* cbuf_display_lock_frame_by_idx (CBuf* cbuf, int frame_no);
Frame* cbuf_display_lock_next_frame (CBuf* cbuf);
Frame* cbuf_get_next_writable_frame (CBuf* cbuf);
void cbuf_reset_queue (CBuf* cbuf);
void cbuf_display_lock_release (CBuf* cbuf);
void cbuf_internal_grab_rewind (CBuf* cbuf);
Frame* cbuf_internal_grab_next_frame (CBuf* cbuf);
Frame* cbuf_display_lock_internal_grab (CBuf* cbuf);
void cbuf_append_waiting (CBuf* cbuf, Frame* new_frame);
void cbuf_mark_frame_written (CBuf* cbuf, Frame* frame);
#endif

#endif
