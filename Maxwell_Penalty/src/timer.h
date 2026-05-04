#ifndef TIMERS_H
#define TIMERS_H
int register_timer(const char *name);
int start_timer(const int timer_id);
int stop_timer(const int timer_id);
void print_timers(void);

#define BEGIN_TIMER(_timer)                                                    \
    start_timer(_timer);                                                       \
    {

#define END_TIMER(_timer)                                                      \
    }                                                                          \
    stop_timer(_timer);

#endif
