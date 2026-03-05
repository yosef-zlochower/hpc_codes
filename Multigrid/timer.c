#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> // needed for timing
#include <unistd.h>

/******************************************************************
* Purpose: Return the current wall-clock time in seconds using gettimeofday.
* Input Variables:
*     (none)
* Output Variables:
*     (none)
* Return Values and indicators of success / failure
*     double, current time in seconds since the Unix epoch
*******************************************************************/
static double get_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return (tv.tv_sec) + 1.0e-6 * tv.tv_usec;
}

#define NAME_LENGTH 1024
#define NUM_TIMERS 20
struct timer
{
    char name[NAME_LENGTH];
    int active;
    int currently_timing;
    double total_time;
    double tstart;
    double t_min;
    double t_max;
    int calls;
};

static int timers_initialized = 0;

static struct timer timers[NUM_TIMERS];

static int active_timers = 0;

/******************************************************************
* Purpose: Zero-initialise the global timers array and mark the system as
*     initialised. Must be called exactly once; calls exit(-1) if called
*     again.
* Input Variables:
*     (none)
* Output Variables:
*     (internal timers[] array is reset)
* Return Values and indicators of success / failure
*     0
*******************************************************************/
static int initialize_timers(void)
{
    if (timers_initialized != 0)
    {
        fprintf(stderr, "initialize_timers should only be called once\n");
        exit(-1);
    }
    for (int i = 0; i < NUM_TIMERS; i++)
    {
        timers[i].name[0] = '\0';
        timers[i].active = 0;
        timers[i].currently_timing = 0;
        timers[i].total_time = 0;
        timers[i].tstart = nan("");
        timers[i].calls = 0;
        timers[i].t_max = -1.0e99;
        timers[i].t_min = +1.0e99;
    }
    timers_initialized = 1;
    return 0;
}

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
int register_timer(const char *name)
{
    if (timers_initialized == 0)
    {
        initialize_timers();
    }
    if (strlen(name) >= NAME_LENGTH)
    {
        fprintf(stderr, "Timer name too long\n");
        exit(-1);
    }

    strncpy(timers[active_timers].name, name, NAME_LENGTH - 1);
    timers[active_timers].name[NAME_LENGTH - 1] = '\0';

    int timer_id = active_timers;
    active_timers++;
    timers[timer_id].currently_timing = 0;
    timers[timer_id].active = 1;
    return timer_id;
}

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
int start_timer(const int timer_id)
{
    if (timers[timer_id].active == 0 || timers[timer_id].currently_timing == 1)
    {
        fprintf(stderr, "timer not in a valid state to start\n");
        exit(-1);
    }
    timers[timer_id].currently_timing = 1;
    timers[timer_id].tstart = get_time();
    timers[timer_id].calls++;

    return 0;
}

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
int stop_timer(const int timer_id)
{
    if (timers[timer_id].active == 0 || timers[timer_id].currently_timing == 0)
    {
        fprintf(stderr, "timer not in a valid state to stop\n");
        exit(-1);
    }
    timers[timer_id].currently_timing = 0;
    const double tend = get_time();
    const double delta_t = tend - timers[timer_id].tstart;

    timers[timer_id].total_time += delta_t;
    if (delta_t > timers[timer_id].t_max)
    {
        timers[timer_id].t_max = delta_t;
    }

    if (delta_t < timers[timer_id].t_min)
    {
        timers[timer_id].t_min = delta_t;
    }

    timers[timer_id].tstart = nan("");

    return 0;
}

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
void print_timers(void)
{
    for (int i = 0; i < NUM_TIMERS; i++)
    {
        if (timers[i].active)
        {
            printf("%s (times total, min, max) %f (%f %f) Calls: %d\n",
                   timers[i].name, timers[i].total_time, timers[i].t_min,
                   timers[i].t_max, timers[i].calls);
        }
    }
}
