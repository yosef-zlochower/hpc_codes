#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h> // needed for timing
#include <unistd.h>

// get_time will return a double containing the current time in
// seconds.
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
