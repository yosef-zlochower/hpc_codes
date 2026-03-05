#ifndef TIMERS_H
#define TIMERS_H
/******************************************************************
* Purpose: Register a new named timer, initialising it to zero elapsed
*     time. Must be called before start_timer. The timer system
*     auto-initialises on the first call to register_timer.
* Input Variables:
*     name: const char*, human-readable name for the timer, must be
*           shorter than 1024 characters
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     timer_id: int, non-negative integer used to identify this timer in
*     subsequent calls. Calls exit(-1) on error (name too long).
*******************************************************************/
int register_timer(const char *name);
/******************************************************************
* Purpose: Start timing for the timer identified by timer_id. Records the
*     current wall-clock time and increments the call count.
* Input Variables:
*     timer_id: int, identifier returned by register_timer
* Output Variables:
*     (none — updates internal timer state)
* Return Values and indicators of success / failure
*     0 on success. Calls exit(-1) if the timer is not in a valid state to
*     start (inactive or already running).
*******************************************************************/
int start_timer(const int timer_id);
/******************************************************************
* Purpose: Stop timing for the timer identified by timer_id. Accumulates
*     elapsed time and updates min/max per-call statistics.
* Input Variables:
*     timer_id: int, identifier returned by register_timer
* Output Variables:
*     (none — updates internal timer state)
* Return Values and indicators of success / failure
*     0 on success. Calls exit(-1) if the timer is not in a valid state to
*     stop (inactive or not currently running).
*******************************************************************/
int stop_timer(const int timer_id);
/******************************************************************
* Purpose: Print timing statistics (total, min, max elapsed time, and call
*     count) for all active timers to stdout.
* Input Variables:
*     (none)
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     void
*******************************************************************/
void print_timers(void);

#define BEGIN_TIMER(_timer)                                                    \
    start_timer(_timer);                                                       \
    {

#define END_TIMER(_timer)                                                      \
    }                                                                          \
    stop_timer(_timer);

#endif
